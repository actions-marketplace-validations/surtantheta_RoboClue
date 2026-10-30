"""Salvage decisions for interrupted native LeRobot v3 recordings.

An interrupted ``lerobot-record`` session leaves states that the pinned LeRobot either refuses with an
unrelated error (footer-less Parquet, unflushed episode metadata) or loads silently (rows of an episode
whose save never finished). This module reads bytes only, builds a per-episode ledger from
``meta/info.json``, the episode-metadata rows, the data-row identity columns and the staging/encoder
leftovers, decides which episodes are complete and usable, and describes a bounded metadata-only salvage
plan. Nothing is written and no original byte is modified; the plan is a description to apply to a copy.

Detection of individual contradictions already lives in ``windows.py`` and ``video_index.py``; findings
here are salvage decisions and cite the existing codes that report the same condition (``detected_by``).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
from pathlib import Path, PurePosixPath

from .artifacts import LEROBOT_REVISION, MAX_JSON_BYTES, InputError, dataset_root, sha256
from .windows import (
    DATA_PATH,
    MAX_FOOTER_BYTES,
    _FooterRowCountConflict,
    _Intake,
    _integer,
    _Unknown,
    bound_artifacts,
)

KIND = 'recording_salvage'
PRIORITY = ('error', 'warning', 'unknown', 'pass')
LEROBOT = f'LeRobot {LEROBOT_REVISION[:7]}'
MAX_ROWS = 1_000_000
MAX_EPISODES = 100_000
MAX_LISTED = 32
MAX_EVIDENCE = 4
MAX_LEFTOVER_ENTRIES = 20_000
MAX_HASHED_LEFTOVER_BYTES = 64 * 1024 * 1024
MAX_VIDEO_HEADERS = 512
PLAN_LIST_LIMIT = 1024  # plan lists are complete up to this many entries; files.lists_complete says so
RESPONSE_BYTES = 192 * 1024
TEMP_DIR = re.compile(r'tmp[a-z0-9_]{8}')  # tempfile.mkdtemp names (dataset_writer.py:81, video_utils.py:1000)
SHARD = re.compile(r'chunk-([0-9]{3,})/file-([0-9]{3,})\.parquet')
SCOPE = ('Bytes of one local native LeRobot v3 dataset: info/stats JSON validity, Parquet footers, the task table size '
         'against the largest task_index of the rows, episode-metadata rows, data-row identity columns (index, '
         'episode_index, frame_index), staging and encoder leftovers and optional MP4 header boxes. The salvage plan is a '
         'metadata-only description for a separate copy; nothing is written. No Parquet page is parsed without a footer, '
         'no video is decoded. Complete stable numeric cohorts additionally compare saved mean/std/min/max with '
         'their scalar/vector rows unless explicitly disabled; image sampling and quantiles are excluded. No recording intent '
         'is inferred. Recoverability of episodes without metadata rests on encoder temp file names with their MP4 header '
         'sample counts, on staging frame file counts, or, for committed episodes beside an existing videos/ '
         'directory, on the recorder writing unbatched video before metadata (their frames are not located in the '
         'video files); frames are neither decoded nor opened, so verify they decode '
         'before relying on a recovery. No symlink is followed: a link anywhere the check reads or scans (meta/, data/, '
         'videos/ and the video files it references or lists, images/ and tmp* encoder directories) stops the check '
         'with the link and its target named, and so does a FIFO, socket or device where a file would be read. '
         'A bounded inventory names all links in those trees without following their targets. A suggested copy '
         'preserves links inside its target; it never recursively follows them.')
LOC = {
    'rows_before_metadata': 'src/lerobot/datasets/dataset_writer.py:326,377-378',
    'footer_at_close': 'src/lerobot/datasets/dataset_writer.py:489-494,649-653; '
                       'src/lerobot/datasets/dataset_metadata.py:175-200; src/lerobot/datasets/lerobot_dataset.py:557',
    'metadata_buffered': 'src/lerobot/datasets/dataset_metadata.py:637-641',
    'totals_per_episode': 'src/lerobot/datasets/dataset_metadata.py:675-680',
    'hub_fallback': 'src/lerobot/datasets/dataset_metadata.py:144-151',
    'every_shard': 'src/lerobot/datasets/io_utils.py:76,83',
    'positional': 'src/lerobot/datasets/dataset_reader.py:304,434',
    'sampler_all_rows': 'src/lerobot/datasets/sampler.py:81-84; src/lerobot/scripts/lerobot_train.py:380-388',
    'len_total_frames': 'src/lerobot/datasets/dataset_reader.py:204-208',
    'resume_labels': 'src/lerobot/datasets/dataset_writer.py:157,287',
    'resume_spans': 'src/lerobot/datasets/dataset_metadata.py:588-601; '
                    'src/lerobot/datasets/dataset_writer.py:450-460,528-545',
    'skip_when_zero': 'src/lerobot/datasets/dataset_metadata.py:221-222',
    'resume_from_zero': 'src/lerobot/datasets/dataset_writer.py:450-454,528-545; '
                        'src/lerobot/datasets/dataset_metadata.py:588-604',
    'task_lookup': 'src/lerobot/datasets/dataset_reader.py:529',
    'load_stats': 'src/lerobot/datasets/io_utils.py:162-176',
    'json_in_place': 'src/lerobot/utils/io_utils.py:48-50',
    'staging': 'src/lerobot/datasets/dataset_writer.py:202-208,257-265',
    'staging_glob': 'src/lerobot/datasets/video_utils.py:529-532',
    'temp_dirs': 'src/lerobot/datasets/dataset_writer.py:81; src/lerobot/datasets/video_utils.py:1000',
    'concat_move': 'src/lerobot/datasets/video_utils.py:758-792',
    'load_metadata': 'src/lerobot/datasets/dataset_metadata.py:218-223',
    'next_file': 'src/lerobot/datasets/utils.py:329-335',
    'video_before_metadata': 'src/lerobot/datasets/dataset_writer.py:328-378',
    'batched_video_after': 'src/lerobot/datasets/dataset_writer.py:380-386',
}
REFERENCE = {'repository': 'https://github.com/huggingface/lerobot', 'revision': LEROBOT_REVISION, 'locations': LOC}
CAUSES_HARD = ('the recording process was killed (SIGKILL, SIGTERM, OOM killer or power loss) before finalize(), '
               'the disk filled while the file was being closed, or a copy or upload was truncated')
CAUSES_SAVE = ('save_episode() stopped after writing the rows (Ctrl-C or an exception during video encoding or '
               'concatenation, a full disk, an encoder error), another writer or converter stopped mid-episode, or '
               'rows were appended by a separate tool')
EXISTING = {  # existing detector codes that report the same underlying condition
    'rows_without_metadata': ['episode.rows_without_metadata', 'episode.declared_totals'],
    'metadata_not_flushed': ['episode.rows_without_metadata'],
    'metadata_unreadable_or_unflushed': ['episode.rows_without_metadata', 'episode.coverage_unknown'],
    'metadata_without_rows': ['episode.metadata_without_rows'],
    'duplicate_metadata': ['episode.metadata_duplicate'],
    'metadata_position': ['episode.metadata_duplicate', 'episode.coverage_unknown'],
    'span_rows_mismatch': ['episode.length_mismatch', 'episode.range_contradiction', 'episode.duplicate_global_index'],
    'span_length': ['episode.range_contradiction'],
    'span_past_rows': ['episode.metadata_without_rows', 'episode.range_contradiction'],
    'ambiguous_rows': ['episode.duplicate_global_index', 'episode.length_mismatch'],
    'empty_episode': ['episode.too_short'],
    'video_interval_past_end': ['video.interval_past_end'],
    'video_unindexed': ['video.feature_without_index'],
    'video_file_missing': ['video.file_missing'],
}
REASON_TEXT = {
    'rows_without_metadata': ('data rows carry this episode index, but no episode-metadata row describes them and '
                              'info.json does not count it'),
    'metadata_not_flushed': ('info.json counts it as saved and its rows sit in files with intact footers, but its '
                             'buffered episode-metadata row was never written'),
    'metadata_unreadable_or_unflushed': ('info.json counts it as saved and its rows sit in files with intact footers, '
                                         'but no readable metadata row describes it: its row was never written or sits '
                                         'in a metadata file that lost its footer'),
    'metadata_without_rows': 'an episode-metadata row exists but no data row carries this episode index',
    'duplicate_metadata': 'more than one episode-metadata row claims this episode index',
    'metadata_position': 'its episode-metadata row is not at the row position LeRobot looks it up by',
    'span_length': 'dataset_to_index - dataset_from_index differs from the declared length',
    'span_past_rows': 'the declared span extends past the readable data rows',
    'rows_in_unreadable_file': 'its data-file pointer names a footer-less file, so its rows cannot be read or verified',
    'span_rows_mismatch': 'the rows at the declared span are not this episode\'s rows in frame order',
    'ambiguous_rows': ('rows carrying its episode index continue past its span end, or its metadata names another data '
                       'file while other rows carry the same index, so the span may serve another recording\'s rows'),
    'empty_episode': 'the declared length is not a positive integer',
    'video_interval_past_end': 'it needs video frames past the end of its video file',
    'video_unindexed': 'a declared video feature has no location for it',
    'video_file_missing': 'its video file is absent while that camera directory exists',
}


def _status(values):
    return next((s for s in PRIORITY if s in values), 'pass')


def _ranges(values, limit=64):
    """Compress integers into sorted [first, last] pairs; bounded."""
    out = []
    for value in sorted(set(values)):
        if out and value == out[-1][1] + 1:
            out[-1][1] = value
        else:
            out.append([value, value])
    return out[:limit]


def _evidence(artifact, pointer, value):
    return {'path': artifact['path'], 'sha256': artifact['sha256'], 'pointer': pointer, 'value': value}


def _next(chunk, file, chunks_size):
    """update_chunk_file_indices (src/lerobot/datasets/utils.py:329-335)."""
    return (chunk + 1, 0) if file == chunks_size - 1 else (chunk, file + 1)


def _video_rel(template, key, chunk, file):
    """The relative video path LeRobot derives (dataset_metadata.py:356-362); unsupported or escaping templates stop
    the check as unknown instead of raising."""
    from .video_index import _relative_video_path
    try:
        return _relative_video_path(template, key, chunk, file)
    except (KeyError, IndexError, ValueError, AttributeError, TypeError) as exc:
        raise _Unknown(f'info.json video_path {str(template)[:80]!r} is not a supported, root-contained template '
                       f'({type(exc).__name__}: {str(exc)[:80]})') from exc


def _safe_relative(relative):
    path = PurePosixPath(relative)
    if (not relative or '\\' in relative or path.is_absolute() or str(path) != relative
            or any(part in ('.', '..') for part in path.parts)):
        raise _Unknown(f'A salvage plan path would leave the dataset root: {relative[:120]!r}')
    return relative


def _kind(mode):
    """What a directory entry that is not a regular file is, in plain words."""
    for test, name in ((stat.S_ISDIR, 'a directory'), (stat.S_ISFIFO, 'a FIFO (named pipe)'), (stat.S_ISSOCK, 'a socket'),
                       (stat.S_ISCHR, 'a character device'), (stat.S_ISBLK, 'a block device'),
                       (stat.S_ISLNK, 'a symlink')):
        if test(mode):
            return name
    return 'not a regular file'


class _StrictIntake(_Intake):
    """The bounded native intake with every refusal naming what it refuses. No symlink is followed: a link at any
    component of a path the check reads or lists stops the check with the link, its target and a remedy that works for
    that target, and no FIFO, socket or device is ever opened. Nothing outside the dataset root is read."""

    def refuse_link(self, link):
        relative = link.relative_to(self.root)
        try:
            written = os.readlink(link)
        except OSError:
            written = '?'
        try:
            target = link.resolve(strict=True)  # a loop raises RuntimeError up to Python 3.12, OSError from 3.13
            mode = target.stat().st_mode
        except (OSError, RuntimeError):
            target = mode = None
        head = f'Refused before reading: {relative} is a symlink to '
        if target is None:
            raise InputError(head + f'{written}, which cannot be followed (a symlink loop or a missing target), so LeRobot '
                             'cannot read through it either and `cp -rL` fails on it too. Restore what it should point to, '
                             'or replace the link with the real file or directory, then rerun.')
        where = 'inside' if target.is_relative_to(self.root) else 'outside'
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise InputError(head + f'{target}, {where} the dataset root, which is {_kind(mode)}, not a regular file or '
                             'directory. Replace the link with the real data file, then rerun.')
        if link.parent.is_relative_to(target):
            raise InputError(head + f'{target}, {where} the dataset root, which contains '
                             + ('the dataset itself' if self.root.is_relative_to(target) else 'the link itself')
                             + ', so a copy that follows the link would recurse into itself. Replace the link with the '
                             'data it should hold (move that data inside the dataset in its place), then rerun.')
        # The remedy acts on this link only: a copy of the whole root would also follow root-level links the check
        # never reads (and fail on cyclic ones).
        temporary = link.with_name(link.name + '.tmp')
        temporary_arg, link_arg = shlex.quote(str(temporary)), shlex.quote(str(link))
        command = (f'test ! -e {temporary_arg} && test ! -L {temporary_arg} && '
                   f'cp -rP -- {shlex.quote(str(target))} {temporary_arg} && rm -- {link_arg} && '
                   f'mv -T -- {temporary_arg} {link_arg}')
        caution = (' The temporary destination must not exist, even as a dangling link; if a previous copy failed, '
                   'inspect and move its temporary output aside before retrying. This copy preserves any links inside '
                   'the target instead of following them; the next check names those links for review.')
        if where == 'inside':
            raise InputError(head + f'{target}, inside the dataset root. Links between dataset files alias one file under '
                             'two names and are not followed; replace the link with a real copy of what it points to '
                             f'(`{command}`), then rerun.' + caution)
        raise InputError(head + f'{target}, outside the dataset root. LeRobot follows such links, but this check reads only '
                         'regular files and directories inside the dataset root and follows no link. Replace the link with '
                         f'a real copy of its target, which LeRobot reads as the same bytes (`{command}`); then rerun.'
                         + caution)

    def refuse_all_links(self):
        """Inventory the dataset trees used by this check without following links or opening file contents.

        An external link's target is deliberately not traversed. Copy remedies preserve its inner links; those are
        accounted for in the next inventory once they are inside the dataset. Unrelated root-level links stay unused.
        """
        links, pending, entries = [], [], 0

        def queue(path):
            nonlocal entries
            entries += 1
            if entries > MAX_LEFTOVER_ENTRIES:
                raise InputError(f'Link inventory exceeds {MAX_LEFTOVER_ENTRIES} entries at '
                                 f'{path.relative_to(self.root)}; no file contents were read. Inspect the tree in parts.')
            pending.append(path)

        with os.scandir(self.root) as children:
            for entry in children:
                if entry.name in ('meta', 'data', 'videos', 'images') or entry.name.startswith('tmp'):
                    queue(Path(entry.path))
        while pending:
            path = pending.pop()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                links.append(path)
            elif stat.S_ISDIR(mode):
                with os.scandir(path) as children:
                    for entry in children:
                        queue(Path(entry.path))
        if links:
            links.sort()
            try:
                self.refuse_link(links[0])
            except InputError as exc:
                inventory = '; '.join(f'{p.relative_to(self.root)} -> {os.readlink(p)}' for p in links)
                raise InputError(f'{exc} All {len(links)} link(s) found in the inspected dataset trees '
                                 f'(targets not followed): {inventory}.') from exc

    def safe(self, path):
        if not path.is_relative_to(self.root):
            raise InputError(f'A dataset path lies outside the dataset root: {path}')
        current = self.root
        for part in path.relative_to(self.root).parts:
            if part == '..':
                raise InputError(f'A dataset path leaves the dataset root: {path.relative_to(self.root)}')
            current /= part
            if current.is_symlink():  # lstat only: nothing is resolved before the link is refused
                self.refuse_link(current)
        return path

    def regular(self, path, *, directory_ok=False):
        """Refuse, by name, anything but a regular file before it is opened: opening a FIFO blocks and a device never
        ends. A directory passes only when the caller handles it."""
        self.safe(path)
        mode = os.lstat(path).st_mode
        if stat.S_ISREG(mode) or (directory_ok and stat.S_ISDIR(mode)):
            return path
        raise InputError(f'Refused before reading: {path.relative_to(self.root)} is {_kind(mode)}, not a regular file. '
                         'LeRobot opens it as a file; restore the real file (from the source or a backup), then rerun.')

    def artifact(self, path, role):
        if os.path.lexists(path):
            self.regular(path)
        return super().artifact(path, role)

    def _entry(self):
        self.entries += 1
        if self.entries > 8192:
            raise _Unknown('Shard discovery entry budget exceeded')

    def discover(self, base):
        """The native shard discovery (windows.py), naming what it refuses; a FIFO named like a shard is never opened."""
        self.safe(base)
        if not base.is_dir():
            raise _Unknown(f'Missing native shard directory: {base.relative_to(self.root)}')
        paths = []
        with os.scandir(base) as chunks:
            for chunk in chunks:
                self._entry()
                directory = self.safe(Path(chunk.path))
                if not chunk.is_dir(follow_symlinks=False):
                    continue
                if not re.fullmatch(r'chunk-[0-9]{3,}', chunk.name):
                    raise _Unknown(f'Nonstandard native shard directory {directory.relative_to(self.root)} is unsupported')
                with os.scandir(directory) as files:
                    for item in files:
                        self._entry()
                        path = self.safe(Path(item.path))
                        if path.suffix != '.parquet':
                            continue
                        self.regular(path, directory_ok=True)
                        if not item.is_file(follow_symlinks=False) or not re.fullmatch(r'file-[0-9]{3,}\.parquet', item.name):
                            raise _Unknown(f'Nonstandard native Parquet entry {path.relative_to(self.root)} is unsupported')
                        paths.append(path)
                        if len(paths) > self.max_files:
                            raise _Unknown('Native shard discovery file budget exceeded')
        return sorted(paths)


class _Work:
    """Bounded, root-contained intake shared by every step of one check."""

    def __init__(self, root, max_rows, max_files, video_headers):
        self.root, self.max_rows, self.max_files, self.video_headers = root, max_rows, max_files, video_headers
        self.intake = _StrictIntake(root, max_rows, max_files)
        self.findings = []
        self.coverage = {'max_rows': max_rows, 'max_files': max_files, 'complete': False}

    def path(self, relative):
        """A root-relative path, checked component by component before anything is read."""
        return self.intake.safe(self.root / _safe_relative(relative))

    def present(self, path):
        """Whether anything sits at path, a dangling link included; a link there is refused by name."""
        if not os.path.lexists(path):
            return False
        self.intake.safe(path)
        return True

    def artifact(self, path, role):
        return self.intake.artifact(self.intake.safe(path), role)

    def json_document(self, relative, role):
        """(document, problem, artifact); a missing file returns (None, None, None)."""
        path = self.path(relative)
        if not path.exists():
            return None, None, None
        self.intake.regular(path, directory_ok=True)
        if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
            raise _Unknown(f'{relative} is not a bounded regular file')
        artifact = self.artifact(path, role)
        raw = path.read_bytes()
        try:
            document = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, {'kind': 'invalid_json', 'detail': str(exc)[:160], 'size': len(raw)}, artifact
        if not isinstance(document, dict):
            return None, {'kind': 'not_an_object', 'detail': type(document).__name__, 'size': len(raw)}, artifact
        return document, None, artifact

    def footer(self, path):
        """Separate proven frame damage from unsupported encryption or reader budgets."""
        size = self.intake.regular(path).stat().st_size
        with path.open('rb') as stream:
            head = stream.read(4)
            if head == b'PARE':
                self.artifact(path, 'encrypted_parquet_unassessed')
                raise _Unknown('Encrypted Parquet footer requires keys and is not assessed by salvage')
            if size < 12:
                return {'kind': 'too_short', 'size': size, 'head_magic': head == b'PAR1', 'tail_magic': False}
            stream.seek(-8, 2)
            tail = stream.read(8)
        length = int.from_bytes(tail[:4], 'little')
        if head != b'PAR1' or tail[4:] != b'PAR1' or not 0 < length <= size - 12:
            return {'kind': 'footer_missing' if tail[4:] != b'PAR1' else 'footer_invalid', 'size': size,
                    'head_magic': head == b'PAR1', 'tail_magic': tail[4:] == b'PAR1', 'last_bytes_hex': tail[4:].hex()}
        if length > MAX_FOOTER_BYTES:
            self.artifact(path, 'parquet_footer_over_budget')
            raise _Unknown('Parquet footer metadata budget exceeded; this does not establish file damage')
        from .native_parquet_guard import UnsupportedParquetMetadata, require_unencrypted_parquet

        try:
            require_unencrypted_parquet(path, max_footer=MAX_FOOTER_BYTES)
        except UnsupportedParquetMetadata as exc:
            self.artifact(path, 'parquet_metadata_unassessed')
            raise _Unknown(str(exc)) from exc
        return None

    def shards(self, base):
        return self.intake.discover(base) if self.present(base) else []

    def table(self, path, columns, role, optional=()):
        import pyarrow as pa
        pf, artifact = self.intake.parquet(path, tuple(columns), role)
        names = pf.schema_arrow.names
        extra = [c for c in optional if c in names and (pa.types.is_integer(pf.schema_arrow.field(c).type)
                                                         or pa.types.is_null(pf.schema_arrow.field(c).type))]
        table = pf.read(columns=list(columns) + extra)
        for name in columns:
            if table.column(name).null_count:
                raise _Unknown(f'Null values in required integer column {name}')
        return table, artifact

    def task_rows(self, path):
        """Row count of a readable meta/tasks.parquet from its footer (the task strings are not read)."""
        import pyarrow.parquet as pq
        self.artifact(path, 'dataset_tasks')
        try:
            return pq.ParquetFile(path).metadata.num_rows
        except (OSError, ValueError) as exc:
            raise _Unknown(f'meta/tasks.parquet footer could not be read: {exc}') from exc

    def leftovers(self):
        """Staging frames under images/ and encoder temp directories in the dataset root (bounded). Nothing in what is
        scanned is skipped silently: a link stops the check with the link named, and a FIFO, socket or device where a
        frame is counted or a temp file read is refused by name."""
        staging, temps, entries = {}, [], 0

        def listing(directory, links=None):
            """Entries of a scanned directory, by name; links (or those whose name matches ``links``) are refused."""
            nonlocal entries
            found = []
            with os.scandir(directory) as scan:
                for entry in scan:
                    entries += 1
                    if entries > MAX_LEFTOVER_ENTRIES:
                        raise _Unknown('Recording leftover entry budget exceeded')
                    if entry.is_symlink() and (links is None or links(entry.name)):
                        self.intake.refuse_link(Path(entry.path))
                    found.append(entry)
            return sorted(found, key=lambda e: e.name)

        images = self.root / 'images'
        if self.present(images) and images.is_dir():
            for key in listing(images):
                if not key.is_dir(follow_symlinks=False):
                    continue
                for episode in listing(key.path):
                    match = re.fullmatch(r'episode-([0-9]{6,})', episode.name)
                    if not match or not episode.is_dir(follow_symlinks=False):
                        continue
                    frames = []
                    for item in listing(episode.path):
                        if not re.fullmatch(r'frame-[0-9]{6}\.(png|tiff)', item.name):
                            continue
                        if item.is_file(follow_symlinks=False):
                            frames.append(item.name)
                        else:
                            self.intake.regular(Path(item.path), directory_ok=True)
                    staging.setdefault(int(match.group(1)), {})[key.name] = {
                        'directory': episode.path, 'frames': len(frames),
                        'first_last': [min(frames), max(frames)] if frames else []}
        # Only entries named like an encoder temp directory are used from the dataset root, so only their links stop it.
        for entry in listing(self.root, links=TEMP_DIR.fullmatch):
            if not TEMP_DIR.fullmatch(entry.name) or not entry.is_dir(follow_symlinks=False):
                continue
            files = []
            for item in listing(entry.path)[:16]:
                if item.is_dir(follow_symlinks=False):
                    continue  # the encoder writes files into its temp directory; a subdirectory is not its output
                path = self.intake.regular(Path(item.path))
                record = {'path': str(path), 'size_bytes': path.stat().st_size}
                if record['size_bytes'] <= MAX_HASHED_LEFTOVER_BYTES:
                    record['sha256'] = sha256(path)
                if path.suffix == '.mp4' and self.video_headers:
                    record['mp4_header'] = self.mp4(path)
                files.append(record)
            temps.append({'directory': entry.path, 'files': files})
            if len(temps) >= MAX_LISTED:
                break
        return staging, temps

    def mp4(self, path):
        from .video_index import _LocalMedia, header_timing
        self.intake.regular(path, directory_ok=True)  # outside the try: a refusal must not read as 'unavailable'
        try:
            media = _LocalMedia(path)
            try:
                return header_timing(media)
            finally:
                media.close()
        except (OSError, ValueError) as exc:
            return {'status': 'unavailable', 'reason': str(exc)[:160]}

    def video_samples(self, keys):
        """Per camera: MP4 header sample totals of the local video files (header boxes only, bounded). The camera
        directory is listed entry by entry in both header modes, since it is the basis for recovering committed episodes
        without metadata: a link there, or a FIFO, socket or device named like a video file, is refused by name (glob
        would follow links and silently skip looping ones). Headers are read only with video_headers."""
        out = {}
        for key in keys:
            base = self.root / 'videos' / key
            if not self.present(base) or not base.is_dir():
                continue
            files = []
            with os.scandir(base) as chunks:
                for chunk in sorted(chunks, key=lambda e: e.name):
                    if chunk.is_symlink():
                        self.intake.refuse_link(Path(chunk.path))
                    if not chunk.name.startswith('chunk-') or not chunk.is_dir(follow_symlinks=False):
                        continue
                    with os.scandir(chunk.path) as items:
                        for item in items:
                            if item.is_symlink():
                                self.intake.refuse_link(Path(item.path))
                            if item.name.startswith('file-') and item.name.endswith('.mp4'):
                                files.append(self.intake.regular(Path(item.path), directory_ok=True))
            if not self.video_headers:
                continue
            files = sorted(files)[:64]
            timings = [self.mp4(path) for path in files]
            parsed = [t for t in timings if t.get('status') == 'parsed']
            out[key] = {'files': len(files), 'parsed': len(parsed), 'samples': sum(t['sample_count'] for t in parsed)}
        return out

    def video_inventory(self):
        """Inventory actual video-tree files without decoding or following links.

        A copy plan must account for unreferenced video shards too. In particular,
        an orphan next-file target left behind by a full-directory copy can be
        overwritten on resume even after its data shard was omitted.
        """
        base = self.path('videos')
        if not self.present(base):
            return []
        self.intake.regular(base, directory_ok=True)
        if not base.is_dir():
            raise _Unknown('videos/ is not a directory')
        pending, files, entries = [base], [], 0
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as children:
                for entry in children:
                    entries += 1
                    if entries > MAX_LEFTOVER_ENTRIES:
                        raise _Unknown('Video copy-plan inventory entry budget exceeded')
                    path = self.path(str(Path(entry.path).relative_to(self.root)))
                    self.intake.regular(path, directory_ok=True)
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    else:
                        files.append(str(path.relative_to(self.root)))
                        if len(files) > self.max_files:
                            raise _Unknown('Video copy-plan inventory file budget exceeded')
        return sorted(files)


def _dataset_root(dataset):
    """dataset_root (artifacts.py), refusing a file argument (the meta/info.json alias) that a symlink carries into
    another directory: resolving it would silently check the dataset around the link's target instead of the one
    named. A dataset directory reached through a link stays accepted; links inside the dataset are refused later."""
    given = Path(dataset).expanduser().absolute()
    if os.path.lexists(given):
        mode = given.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            raise InputError(f'Refused before reading: the argument {given} is {_kind(mode)}, not a native dataset '
                             'directory or regular metadata file. Restore the real data and pass the dataset directory.')
        if stat.S_ISLNK(mode) and not given.exists():
            raise InputError(f'Refused before reading: the argument {given} is a symlink to {os.readlink(given)}, '
                             'whose target is missing or loops. Restore the target or pass the real dataset directory.')
    root = dataset_root(given)
    if not given.is_dir() and os.path.lexists(given):
        named = given.parent.parent if given.parent.name == 'meta' else given.parent
        if given.is_symlink() or named.resolve() != root:
            raise InputError(f'Refused before reading: the argument {given} is a file reached through a symlink '
                             f'(it resolves to {given.resolve()}), so it would check {root} instead of {named}. Pass the '
                             'dataset directory itself; any link inside it is then refused by name.')
    return root


_RECOVERY_COPY_RECHECK = object()


def check_salvage(dataset, *, max_rows=200000, max_files=1024, video_headers=True, compare_stats=True,
                  _recovery_copy_token=None) -> dict:
    """Decide which episodes of an interrupted local LeRobot v3 recording are complete, which must be dropped
    and why, and describe a bounded metadata-only salvage plan. Reads bytes only; writes nothing.

    No symlink is followed. A link anywhere the check reads or scans, and a FIFO, socket or device where a file would
    be read, stop the check with an InputError that names it (and a link's target) and says what to do: replace that
    link with a real copy of its target, preserving any links inside that target for review, or restore the real file.
    All links in the bounded dataset trees are named in one refusal; targets are not traversed."""
    if not _integer(max_rows, minimum=1, maximum=MAX_ROWS) or not _integer(max_files, minimum=1, maximum=4096):
        raise InputError(f'Require strict integer max_rows in [1,{MAX_ROWS}] and max_files in [1,4096]')
    if type(video_headers) is not bool:
        raise InputError('video_headers must be a strict boolean')
    if type(compare_stats) is not bool:
        raise InputError('compare_stats must be a strict boolean')
    if not isinstance(dataset, (str, Path)) or not str(dataset) or '://' in str(dataset):
        raise InputError('Salvage checks require an explicit local dataset directory (no network)')
    try:
        import numpy  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise InputError('Install policy-deployment-doctor[dataset] for salvage checks') from exc
    try:
        root = _dataset_root(dataset)
    except InputError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:  # a symlink loop, ~user naming no user, a NUL byte
        raise InputError(f'Cannot resolve the dataset path {str(dataset)[:200]}: {exc}') from exc
    if not root.is_dir():
        raise InputError('Expected a native dataset directory')
    if (os.path.lexists(root / '.roboclue-copy-incomplete.json') and
            _recovery_copy_token is not _RECOVERY_COPY_RECHECK):
        raise InputError('This is an incomplete RoboClue recovery copy, not a verified candidate. Keep it '
                         'unqualified; return to the original dataset and create a new copy in a new directory.')
    work = _Work(root, max_rows, max_files, video_headers)
    result = {'schema_version': 1, 'kind': KIND, 'status': 'unknown', 'counts': {}, 'coverage': work.coverage,
              'findings': work.findings, 'episodes': {}, 'plan': None, 'artifacts': [], 'scope': SCOPE,
              'upstream_reference': REFERENCE}
    try:
        work.intake.refuse_all_links()
        _assess(work, result)
        validity = (result.get('plan') or {}).get('validity', {})
        if (compare_stats and work.coverage.get('complete') is True
                and validity.get('current_directory_resume_status') == 'no_hazard_detected'):
            from .dataset import _SCOPED_STATISTICS_RECHECK, scan_dataset

            row_report = scan_dataset(str(root), max_rows=max_rows, max_files=min(max_files, 1000), compare_stats=True,
                                      _intake_token=_SCOPED_STATISTICS_RECHECK)
            comparison = row_report['statistics_comparison']
            result['statistics_comparison'] = comparison
            work.coverage['observed_row_statistics_compared'] = bool(comparison['compared_features'])
            for finding in row_report['findings']:
                if finding.get('code') == 'dataset.statistics_rows_mismatch':
                    work.findings.append({**finding, 'status': 'error', 'message': finding['detail'],
                        'remedy': 'Recompute statistics for the retained numeric rows into a new copy; '
                                  'preserve the original data and statistics.'})
            if (comparison['status'] == 'unknown' and
                    not any(f.get('code') == 'salvage.stats_missing' for f in work.findings)):
                work.findings.append({'code': 'salvage.statistics_values_unassessed', 'status': 'unknown',
                    'message': 'Saved numeric statistics were not fully compared with the retained rows: '
                               + str(comparison['reason'] or 'unsupported statistic values') + '.',
                    'remedy': 'Supply supported native statistics and a complete bounded cohort before resume.',
                    'evidence': []})
        else:
            result['statistics_comparison'] = {'status': 'unknown', 'compared_features': [],
                'reason': 'Explicitly disabled.' if not compare_stats else
                          'Repair the incomplete or inconsistent recording ledger before comparing numeric values.'}
            work.coverage['observed_row_statistics_compared'] = False
        work.intake.stable()
    except _Unknown as exc:
        if isinstance(exc, _FooterRowCountConflict):
            work.findings.append(dict(exc.finding, code='salvage.parquet_row_counts'))
        work.findings.append({'code': 'salvage.coverage_unknown', 'status': 'unknown',
                              'message': f'The salvage ledger could not be completed: {exc}. No episode is declared '
                                         'complete or droppable from a partial reading.',
                              'remedy': 'Raise the stated budget explicitly or inspect the cited layout, then rerun.',
                              'evidence': []})
        result['plan'] = {'strategy': 'unknown', 'reason': str(exc)}
        work.coverage['complete'] = False
        _stable(work)
    except InputError:
        raise  # refusals already name what they refuse
    except (OSError, ValueError, OverflowError, MemoryError) as exc:
        raise InputError(f'Cannot inspect bounded salvage evidence: {exc}') from exc
    except RuntimeError as exc:  # a path error such as a symlink loop met by a library call; never a raw RuntimeError
        if isinstance(exc, RecursionError):
            raise
        raise InputError(f'Cannot inspect bounded salvage evidence ({type(exc).__name__}): {exc}') from exc
    result['status'] = _status([f['status'] for f in work.findings])
    result['counts']['findings'] = {s: sum(f['status'] == s for f in work.findings) for s in PRIORITY}
    result['artifacts'] = list(work.intake.artifacts.values())
    bound_artifacts(result, str(root), RESPONSE_BYTES)
    if len(json.dumps(result, allow_nan=False).encode()) > RESPONSE_BYTES:
        raise InputError('Salvage response exceeds its bounded transport budget')
    return result


def _stable(work):
    """Re-verify every file read after a partial reading, as after a complete one."""
    try:
        work.intake.stable()
    except InputError:
        raise
    except (OSError, RuntimeError) as exc:
        raise InputError(f'Cannot re-verify the files read: {exc}') from exc


def _intact_metadata_repair(meta_rows, total_episodes, total_frames, spans, idx, lab, frm, unreadable, video):
    """Repair only a complete, unique positional ledger, never repeated/partial runs.

    Old v3 converters left file-local spans and previous-file pointers at
    rollover. The row identities, not the broken spans, must witness the fix.
    A run crossing files or disagreeing with its declared length is refused.
    """
    import numpy as np

    rows = len(lab)
    if (unreadable or video or not meta_rows or len(meta_rows) != total_episodes or rows != total_frames
            or [m['episode_index'] for m in meta_rows] != list(range(total_episodes))
            or not np.array_equal(idx, np.arange(rows))):
        return meta_rows, [], []
    corrected, pointers, windows = [], [], []
    position = 0
    for ep, item in enumerate(meta_rows):
        end = position + item['length']
        if (end <= position or end > rows or not np.all(lab[position:end] == ep)
                or not np.array_equal(frm[position:end], np.arange(item['length']))):
            return meta_rows, [], []
        shard = next(s for s in spans if s['start'] <= position < s['end'])
        if end > shard['end']:
            return meta_rows, [], []
        fixed = dict(item, **{'data/chunk_index': shard['chunk'], 'data/file_index': shard['file_index'],
                             'dataset_from_index': position, 'dataset_to_index': end})
        witness = [_evidence(shard['artifact'], f'row {position - shard["start"]}/episode_index', ep),
                   _evidence(shard['artifact'], f'row {position - shard["start"]}/index', position)]
        for keys, changes in ((('data/chunk_index', 'data/file_index'), pointers),
                              (('dataset_from_index', 'dataset_to_index'), windows)):
            if any(item[k] != fixed[k] for k in keys):
                changes.append({'episode_index': ep, 'from': {k: item[k] for k in keys},
                                'to': {k: fixed[k] for k in keys},
                                'evidence': [_evidence(item['_artifact'], f'row {item["_row"]}/{k}', item[k])
                                             for k in keys] + witness})
        corrected.append(fixed)
        position = end
    # Leave the established pointer-only path unchanged; this branch fixes the
    # complete-ledger span corruption that previously excluded intact episodes.
    return (corrected, pointers, windows) if windows and position == rows else (meta_rows, [], [])


def _metadata_row_conflicts(meta_rows, spans, idx, lab, frm, labelled, first_row, claims,
                            metadata_positions_complete=True):
    """Name contradictions without choosing whether rows or metadata are wrong.

    A readable unique run can contradict a span even when an unrelated defect
    prevents a complete repair. These observations never authorize a rewrite.
    """
    import numpy as np

    observed, span_count, pointer_count, observation_count = [], 0, 0, 0
    physical_count, index_count, span_examples, position_unresolved = 0, 0, [], 0
    duplicate_episodes = sorted(ep for ep, positions in claims.items() if len(positions) > 1)
    wrong_positions = [(p, item) for p, item in enumerate(meta_rows)
                       if metadata_positions_complete and item['episode_index'] != p]
    for ep in duplicate_episodes:
        observation_count += 1
        if len(observed) < MAX_LISTED:
            observed.append({'episode_index': ep, 'metadata_identity_conflict': True,
                'metadata_row_positions': claims[ep][:MAX_LISTED],
                'evidence': [_evidence(meta_rows[p]['_artifact'], f'row {meta_rows[p]["_row"]}/episode_index', ep)
                             for p in claims[ep][:2]]})
    for position, item in wrong_positions:
        observation_count += 1
        if len(observed) < MAX_LISTED:
            observed.append({'episode_index': item['episode_index'], 'metadata_identity_conflict': True,
                'metadata_row_position': position,
                'evidence': [dict(_evidence(item['_artifact'], f'row {item["_row"]}/episode_index',
                                           item['episode_index']), expected_row_position=position)]})
    for item in meta_rows:
        ep, count = item['episode_index'], labelled.get(item['episode_index'], 0)
        if len(claims.get(ep, ())) != 1 or count < 1 or count != item['length']:
            continue
        start, end = first_row[ep], first_row[ep] + count
        shard = next(s for s in spans if s['start'] <= start < s['end'])
        if (end > shard['end'] or not np.all(lab[start:end] == ep)
                or not np.array_equal(frm[start:end], np.arange(count))):
            continue
        first_index = int(idx[start])
        contiguous_index = first_index >= 0 and np.array_equal(
            idx[start:end], np.arange(first_index, first_index + count))
        # Native sampling uses physical row positions, not the stored index.
        # A file-local index can agree with broken spans while both disagree
        # with the rows the consumer would actually select.
        declared_span = (item['dataset_from_index'], item['dataset_to_index'])
        position_known = shard.get('position_known', True)
        physical_differs = position_known and declared_span != (start, end)
        index_differs = (not np.array_equal(idx[start:end], np.arange(start, end)) if position_known
                         else not contiguous_index)
        position_unresolved += int(not position_known)
        span_differs = physical_differs or index_differs
        physical_count += int(physical_differs)
        index_count += int(index_differs)
        if span_differs and len(span_examples) < 3:
            span_examples.append(ep)
        pointer_differs = (item['data/chunk_index'], item['data/file_index']) != (shard['chunk'], shard['file_index'])
        span_count += int(span_differs)
        pointer_count += int(pointer_differs)
        observation_count += int(span_differs or pointer_differs or not position_known)
        if (span_differs or pointer_differs or not position_known) and len(observed) < MAX_LISTED:
            observed.append({'episode_index': ep,
                'declared_span': [item['dataset_from_index'], item['dataset_to_index']],
                'observed_index_span': [first_index, first_index + count] if contiguous_index else None,
                'physical_span_conflict': physical_differs, 'row_index_conflict': bool(index_differs),
                'observed_position_span': [start, end] if position_known else None,
                'readable_position_span': [start, end], 'global_position_known': position_known,
                'declared_pointer': [item['data/chunk_index'], item['data/file_index']],
                'observed_pointer': [shard['chunk'], shard['file_index']],
                'evidence': [_evidence(item['_artifact'], f'row {item["_row"]}/dataset_from_index', item['dataset_from_index']),
                             _evidence(shard['artifact'], f'row {start - shard["start"]}/episode_index', ep),
                             _evidence(shard['artifact'], f'row {start - shard["start"]}/index', first_index)]})
    return {'span_conflicts': span_count, 'pointer_conflicts': pointer_count, 'observations': observed,
            'physical_span_conflicts': physical_count, 'row_index_conflicts': index_count,
            'position_unresolved': position_unresolved,
            'unresolved_position_rows': sum(s['end'] - s['start'] for s in spans if not s.get('position_known', True)),
            'metadata_identity_conflicts': len(set(duplicate_episodes)
                                               | {item['episode_index'] for _, item in wrong_positions}),
            'duplicate_metadata_identities': len(duplicate_episodes),
            'metadata_order_conflicts': len(wrong_positions),
            'span_examples': span_examples,
            'observations_total': observation_count, 'observations_complete': observation_count <= MAX_LISTED,
            'scope': 'Bounded readable unique episode identity runs versus declared metadata. '
                     'Observations are not approved corrections or proof of missing data.'}


def _assess(work, result):
    import numpy as np
    root, findings = work.root, work.findings

    # ---- JSON metadata ------------------------------------------------------------------------------
    info, info_problem, info_art = work.json_document('meta/info.json', 'dataset_info')
    saved_stats, stats_problem, stats_art = work.json_document('meta/stats.json', 'dataset_stats')
    broken = [(r, p, a) for r, p, a in (('meta/info.json', info_problem, info_art),
                                        ('meta/stats.json', stats_problem, stats_art)) if p]
    if broken:
        findings.append({'code': 'salvage.file_unreadable', 'status': 'error', 'files': [r for r, _, _ in broken],
            'message': f'{", ".join(r for r, _, _ in broken)} is not a valid JSON object ({broken[0][1]["detail"]}). '
                       f'{LEROBOT} rewrites info.json and stats.json in place after every saved episode '
                       f'({LOC["totals_per_episode"]}; plain open("w") + json.dump, {LOC["json_in_place"]}) and reads both '
                       f'when loading ({LOC["load_metadata"]}), so loading and --resume fail. Possible causes: the disk '
                       'filled during a rewrite, the process was killed mid-write, or a copy was truncated.',
            'remedy': 'Keep the original bytes. Restore the file from a backup or another copy; stats.json can be '
                      'recomputed from the rows, info.json totals cannot be guessed. Rerun this check afterwards.',
            'evidence': [_evidence(a, '', {'size_bytes': p['size'], 'problem': p['kind']}) for _, p, a in broken]})
    if info is None:
        if info_problem is None:
            raise _Unknown('meta/info.json is absent; this is not a native LeRobot dataset root')
        result['plan'] = {'strategy': 'not_metadata_only', 'reason': 'meta/info.json is unreadable'}
        return
    if info.get('storage_format', 'parquet') != 'parquet':
        # Leftover .parquet shards are not the recording under another declared backend.
        raise _Unknown('Only the absent/default or explicitly declared Parquet backend is supported')
    if not str(info.get('codebase_version', '')).startswith('v3.') or info.get('data_path') != DATA_PATH:
        raise _Unknown('Only native v3 metadata with the standard data_path template is supported')
    total_episodes, total_frames = info.get('total_episodes'), info.get('total_frames')
    chunks_size = info.get('chunks_size', 1000)
    if not _integer(total_episodes) or not _integer(total_frames) or not _integer(chunks_size, minimum=1):
        raise _Unknown('info.json total_episodes/total_frames/chunks_size are not valid integers')
    if total_episodes > MAX_EPISODES:
        raise _Unknown(f'Declared episode count exceeds the bounded ledger limit ({MAX_EPISODES})')
    from .statistics_counts import statistics_count_findings

    findings.extend(statistics_count_findings(info, saved_stats, info_artifact=info_art,
                                              stats_artifact=stats_art))
    total_tasks = info.get('total_tasks', 0)
    if not _integer(total_tasks):
        raise _Unknown('info.json total_tasks is not a non-negative integer')
    features = info.get('features') if isinstance(info.get('features'), dict) else {}
    video_keys = sorted(k for k, v in features.items() if isinstance(v, dict) and v.get('dtype') == 'video')
    template = info.get('video_path')
    if video_keys:
        if not isinstance(template, str):
            raise _Unknown('info.json declares video features but no video_path template')
        for key in video_keys:
            _video_rel(template, key, 0, 0)  # fails as unknown before any plan path is derived
        # Camera directories are tested for existence below (missing files, frames of orphan rows); a linked or
        # dangling videos/ or videos/<key> is refused here by name, in both header modes, instead of being followed.
        for relative in ['videos'] + [f'videos/{key}' for key in video_keys]:
            if os.path.lexists(root / relative):
                work.path(relative)
    declared = [_evidence(info_art, '/total_episodes', total_episodes),
                _evidence(info_art, '/total_frames', total_frames)]
    committed = {'total_episodes': total_episodes, 'total_frames': total_frames}

    # ---- Parquet footers ------------------------------------------------------------------------------
    meta_files, data_files = work.shards(root / 'meta' / 'episodes'), work.shards(root / 'data')
    if len(meta_files) + len(data_files) > work.max_files:
        raise _Unknown('Combined native shard budget exceeded')
    tasks = root / 'meta' / 'tasks.parquet'
    unreadable = {}
    for path in ([tasks] if work.present(tasks) else []) + meta_files + data_files:
        problem = work.footer(path)  # refuses a link, FIFO, socket or device by name before opening
        if problem:
            unreadable[path] = (problem, work.artifact(path, 'parquet_without_footer'))
    if unreadable:
        items = sorted(unreadable.items(), key=lambda item: str(item[0]))
        first = items[0][1][0]
        findings.append({'code': 'salvage.file_unreadable', 'status': 'error',
            'files': [str(p.relative_to(root)) for p, _ in items[:MAX_LISTED]],
            'message': f'{len(items)} Parquet file(s) have no valid footer (e.g. {items[0][0].relative_to(root)}: '
                       f'{first["size"]} bytes, starts with PAR1={first["head_magic"]}, ends with PAR1={first["tail_magic"]}). '
                       f'{LEROBOT} writes data and episode metadata through a ParquetWriter that stays open across episodes '
                       f'and writes the footer only when it is closed ({LOC["footer_at_close"]}); meta/tasks.parquet is '
                       f'rewritten whole by pandas when a new task appears. LeRobot reads every one of these files '
                       f'({LOC["every_shard"]}; {LOC["load_metadata"]}) and fails at load (observed natively: ArrowInvalid '
                       '"Parquet magic bytes not found in footer" naming no file for a metadata file; '
                       'DatasetGenerationError for a data file, whose path the datasets library logs at ERROR level). '
                       '--resume fails the same way when a metadata or task file is affected; with only data files '
                       f'affected it proceeds (see salvage.resume_hazard). Possible causes: {CAUSES_HARD}.',
            'remedy': 'Keep these bytes untouched. Rows inside a footer-less file cannot be recovered by a metadata-only '
                      'plan; that needs a byte-level footer rebuild in a separate copy, or re-recording. The plan omits '
                      'footer-less data and metadata files; a footer-less task table must be restored before any copy loads.',
            'evidence': [_evidence(a, '/footer', {k: v for k, v in p.items() if k != 'size'} | {'size_bytes': p['size']})
                         for _, (p, a) in items[:MAX_EVIDENCE]]})
    readable_meta = [p for p in meta_files if p not in unreadable]
    readable_data = [p for p in data_files if p not in unreadable]
    task_count = work.task_rows(tasks) if tasks.exists() and tasks not in unreadable else None
    staging, temps = work.leftovers()

    # ---- episode-metadata rows in LeRobot's positional order (sorted shards, physical rows) --------------
    meta_rows = []
    required = ('episode_index', 'length', 'dataset_from_index', 'dataset_to_index', 'data/chunk_index',
                'data/file_index')
    optional = ('meta/episodes/chunk_index', 'meta/episodes/file_index') + tuple(
        f'videos/{k}/{c}' for k in video_keys for c in ('chunk_index', 'file_index'))
    for path in readable_meta:
        table, artifact = work.table(path, required, 'episode_metadata', optional)
        for row, item in enumerate(table.to_pylist()):
            if any(not _integer(item[c], minimum=-(2 ** 53)) for c in required):
                raise _Unknown('Episode metadata identity/span/pointer values are not integers')
            item.update(_artifact=artifact, _row=row, _file=str(path.relative_to(root)))
            meta_rows.append(item)
            if len(meta_rows) > MAX_EPISODES:
                raise _Unknown('Episode metadata row budget exceeded')

    # ---- data identity columns over the readable shards (the view a salvage copy would give LeRobot) ------
    # Every video file the metadata references must be a regular file inside the root. Refuse a symlink (dangling or
    # not), a FIFO, a device or a directory here, by name and in both header modes, exactly as Parquet and JSON files
    # are refused, so that no such path reaches the existing index check or the plan's copy list.
    referenced_videos = set()
    for item in meta_rows:
        for key in video_keys:
            chunk, file = item.get(f'videos/{key}/chunk_index'), item.get(f'videos/{key}/file_index')
            if _integer(chunk) and _integer(file):
                referenced_videos.add(_video_rel(template, key, chunk, file))
    for relative in sorted(referenced_videos):
        path = root / relative
        if work.present(path):
            work.intake.regular(path)

    import pyarrow.parquet as pq
    spans, parts, rows, max_task = [], [], 0, None
    first_unreadable_data = next((p for p in data_files if p in unreadable), None)
    for path in readable_data:
        table, artifact = work.table(path, ('index', 'episode_index', 'frame_index'), 'dataset_parquet', ('task_index',))
        match = SHARD.fullmatch(str(path.relative_to(root / 'data')))
        spans.append({'file': str(path.relative_to(root)), 'artifact': artifact, 'start': rows,
                      'end': rows + table.num_rows, 'chunk': int(match.group(1)), 'file_index': int(match.group(2)),
                      'position_known': first_unreadable_data is None or path < first_unreadable_data,
                      'row_groups': pq.ParquetFile(path).metadata.num_row_groups})
        parts.append([table.column(c).to_numpy().astype(np.int64) for c in ('index', 'episode_index', 'frame_index')])
        if 'task_index' in table.column_names and table.num_rows:
            column = table.column('task_index')
            if column.null_count == 0:
                largest = int(column.to_numpy().max())
                max_task = largest if max_task is None else max(max_task, largest)
        rows += table.num_rows
    empty = np.zeros(0, np.int64)
    idx, lab, frm = (np.concatenate([p[i] for p in parts]) if parts else empty for i in range(3))
    work.coverage.update(metadata_files_read=len(readable_meta), data_files_read=len(readable_data),
                         metadata_rows_read=len(meta_rows), data_rows_read=rows, files_without_footer=len(unreadable))
    if first_unreadable_data is not None:
        work.coverage['global_data_positions_complete'] = False

    def shard_at(position):
        return next(s for s in spans if s['start'] <= position < s['end'])

    def row_ref(position, column):
        shard = shard_at(position)
        values = {'index': idx, 'episode_index': lab, 'frame_index': frm}[column]
        return _evidence(shard['artifact'], f'row {position - shard["start"]}/{column}', int(values[position]))

    def meta_ref(item, column):
        return _evidence(item['_artifact'], f'row {item["_row"]}/{column}', item.get(column))

    labelled, first_row = {}, {}
    if rows:
        values, starts, counts = np.unique(lab, return_index=True, return_counts=True)
        labelled = {int(v): int(c) for v, c in zip(values, counts, strict=True)}
        first_row = {int(v): int(s) for v, s in zip(values, starts, strict=True)}
    claims = {}
    for position, item in enumerate(meta_rows):
        claims.setdefault(item['episode_index'], []).append(position)
    if len(set(labelled) | set(claims)) > MAX_EPISODES:
        raise _Unknown('Episode count budget exceeded')
    video, unlocatable, truncated = _video_reasons(work, meta_rows, meta_files, unreadable, video_keys, template)
    unreadable_names = {str(p.relative_to(root)) for p in unreadable}
    original_meta_rows = meta_rows
    # A header opt-out cannot prove the same repair as a complete video check.
    # Keep identities as evidence, but withhold automatic metadata corrections.
    repaired_pointers, repaired_spans = [], []
    if not video_keys or work.coverage.get('video_index_status') == 'complete':
        meta_rows, repaired_pointers, repaired_spans = _intact_metadata_repair(
            meta_rows, total_episodes, total_frames, spans, idx, lab, frm, unreadable, video)
    conflicts = None
    blockers = []
    if not repaired_spans:
        conflicts = _metadata_row_conflicts(meta_rows, spans, idx, lab, frm, labelled, first_row, claims,
                                           metadata_positions_complete=not any(p in unreadable for p in meta_files))
        if (conflicts['span_conflicts'] or conflicts['unresolved_position_rows']
                or conflicts['metadata_identity_conflicts']):
            if unreadable:
                blockers.append('unreadable Parquet files')
            if video:
                blockers.append('video inconsistencies')
            if video_keys and work.coverage.get('video_index_status') != 'complete':
                blockers.append('video completeness was not checked')
            if len(meta_rows) != total_episodes or rows != total_frames:
                blockers.append('declared totals disagree with the readable ledger')
            if [m['episode_index'] for m in meta_rows] != list(range(total_episodes)):
                blockers.append('metadata episode identities are not a unique ordered sequence')
            if not np.array_equal(idx, np.arange(rows)):
                blockers.append('global row indices do not match the complete positional ledger')
            if not blockers:
                blockers.append('an episode length, frame order or file boundary prevents a complete identity tiling')
            if conflicts['unresolved_position_rows']:
                blockers.append('global row positions after an unreadable data shard are unknown')
            if conflicts['metadata_identity_conflicts']:
                blockers.append('duplicate metadata identities do not establish a safe exclusion plan')
            findings.append({'code': 'salvage.metadata_row_conflict',
                'status': 'error' if (conflicts['physical_span_conflicts']
                                      or conflicts['metadata_identity_conflicts']) else 'unknown',
                'message': f'{conflicts["span_conflicts"]} episode span(s) disagree with physical rows or stored indices '
                           f'(examples: {conflicts["span_examples"]}); '
                           f'{conflicts["physical_span_conflicts"]} declared span(s) contradict physical episode runs. '
                           f'{conflicts["metadata_identity_conflicts"]} metadata episode identity/order conflict(s) '
                           f'include {conflicts["duplicate_metadata_identities"]} duplicated identity/identities and '
                           f'{conflicts["metadata_order_conflicts"]} rows at incompatible known positions. '
                           f'Global physical positions remain unresolved for {conflicts["position_unresolved"]} '
                           'readable episode run(s); readable-only offsets are not evidence of a contradiction. '
                           f'{conflicts["pointer_conflicts"]} data-file pointer(s) also disagree. '
                           'A complete automatic repair is not established: ' + '; '.join(blockers) + '. '
                           'The disagreement does not establish which representation is wrong or justify exclusions.',
                'remedy': 'Preserve every original file. Review the cited row/metadata contradictions and the other '
                          'findings before deriving a repair. No keep/drop or automatic-copy plan is approved; '
                          'do not resume this directory until the ledger is resolved.',
                'evidence': [e for item in conflicts['observations'][:2] for e in item['evidence']]})
            if video:
                findings.append({'code': 'salvage.video_conflict', 'status': 'error',
                    'episodes': _ranges(video),
                    'message': 'Video references are also inconsistent for episode(s) ' + str(_ranges(video)) + '.',
                    'remedy': 'Review the video references independently; they do not justify discarding other episodes.',
                    'evidence': [e for reasons in list(video.values())[:2] for _, evidence in reasons for e in evidence][:MAX_EVIDENCE]})
            result['metadata_row_conflicts'] = conflicts
        else:
            conflicts = None
    if repaired_spans:
        findings.append({'code': 'salvage.metadata_rows_repairable', 'status': 'error',
            'message': f'The identity columns of {total_episodes} episodes / {rows} rows form unique episode runs: '
                       'frame order and global indices agree with the declared episode lengths. '
                       f'Episode metadata has {len(repaired_pointers)} incorrect data-file pointer(s) and '
                       f'{len(repaired_spans)} incorrect row span(s). This is metadata corruption, not missing rows; '
                       'a converter rollover defect is one possible cause.',
            'remedy': 'Repair the listed metadata pointers and row spans from the data files in a separate copy. '
                      'Keep every episode and preserve the original bytes; no episode needs to be dropped for this '
                      'metadata contradiction. Recheck the copy, including available video and native loading.',
            'evidence': [e for f in (repaired_pointers[:2] + repaired_spans[:1])
                         for e in f['evidence']][:MAX_EVIDENCE]})

    def span_problem(item, ep):
        start, end, length = item['dataset_from_index'], item['dataset_to_index'], item['length']
        if length < 1:
            return 'empty_episode', [meta_ref(item, 'length')]
        if end - start != length:
            return 'span_length', [meta_ref(item, 'length'), meta_ref(item, 'dataset_to_index')]
        pointed = f'data/chunk-{item["data/chunk_index"]:03d}/file-{item["data/file_index"]:03d}.parquet'
        if (ep not in labelled or end > rows) and pointed in unreadable_names:
            return 'rows_in_unreadable_file', [meta_ref(item, 'data/file_index')]
        if ep not in labelled:
            return 'metadata_without_rows', [meta_ref(item, 'episode_index')]
        if start < 0 or end > rows:
            return 'span_past_rows', [meta_ref(item, 'dataset_to_index')]
        window = slice(start, end)
        bad = np.flatnonzero((lab[window] != ep) | (idx[window] != np.arange(start, end))
                             | (frm[window] != np.arange(length)))
        if bad.size:
            return 'span_rows_mismatch', [meta_ref(item, 'dataset_from_index'), row_ref(start + int(bad[0]), 'episode_index'),
                                          row_ref(start + int(bad[0]), 'index')]
        return None, []

    def ambiguous(item, ep):
        """Identity columns cannot tell which recording the span serves when (a) the row right after the span end
        continues the same episode label (an interrupted episode's longer run, e.g. kept by aggregate_datasets), or
        (b) the span's rows sit in another file than the metadata names while other rows carry the same label
        (observed natively after --resume). Returns the evidence or None."""
        start, end = item['dataset_from_index'], item['dataset_to_index']
        if end < rows and int(lab[end]) == ep:
            return [meta_ref(item, 'dataset_to_index'), row_ref(end, 'episode_index'), row_ref(end, 'frame_index')]
        shard = shard_at(start)
        moved = (shard['chunk'], shard['file_index']) != (item['data/chunk_index'], item['data/file_index'])
        if moved and labelled.get(ep, 0) > item['length']:
            return [meta_ref(item, 'data/file_index'), row_ref(start, 'index')]
        return None

    if conflicts:
        index_problems = []
        for shard in spans:
            start, end = shard['start'], shard['end']
            if shard['position_known']:
                positions = np.flatnonzero(idx[start:end] != np.arange(start, end)) + start
            else:
                # A missing predecessor leaves absolute positions unknown, but
                # negative/repeated/gapped local indices are independent facts.
                positions = np.union1d(np.flatnonzero(idx[start:end] < 0),
                                      np.flatnonzero(np.diff(idx[start:end]) != 1) + 1) + start
            index_problems.extend(int(p) for p in positions)
        bad_indices = np.asarray(index_problems, dtype=np.int64)
        bad_frames = []
        for ep, count in labelled.items():
            start = first_row[ep]
            if np.all(lab[start:start + count] == ep) and not np.array_equal(frm[start:start + count], np.arange(count)):
                bad_frames.append(ep)
        if bad_indices.size or bad_frames:
            findings.append({'code': 'salvage.row_identity', 'status': 'error',
                'episodes': _ranges(set(int(lab[p]) for p in bad_indices) | set(bad_frames)),
                'message': f'{int(bad_indices.size)} stored row index value(s) disagree with known physical positions '
                           'or local contiguous order; '
                           f'frame order is inconsistent for episode(s) {_ranges(bad_frames)}. '
                           'These are observed identity contradictions, not an exclusion recommendation.',
                'remedy': 'Preserve the original rows. Compare row identities with the source recording before '
                          'repairing metadata or rewriting indices in a separate copy.',
                'evidence': ([row_ref(int(bad_indices[0]), 'index')] if bad_indices.size else [])
                            + [row_ref(first_row[ep], 'frame_index') for ep in bad_frames[:2]]})

    # ---- keepable prefix: metadata row i must be episode i with a span tiling the rows from 0 -------------
    kept, frames, fixes = 0, 0, list(repaired_pointers)
    while kept < len(meta_rows):
        item = meta_rows[kept]
        if item['episode_index'] != kept or item['dataset_from_index'] != frames or video.get(kept):
            break
        if span_problem(item, kept)[0] or ambiguous(item, kept):
            break
        shard = shard_at(item['dataset_from_index'])
        if conflicts and ((shard['chunk'], shard['file_index']) !=
                             (item['data/chunk_index'], item['data/file_index'])):
            break
        if (shard['chunk'], shard['file_index']) != (item['data/chunk_index'], item['data/file_index']):
            fixes.append({'episode_index': kept,
                          'from': {'data/chunk_index': item['data/chunk_index'], 'data/file_index': item['data/file_index']},
                          'to': {'data/chunk_index': shard['chunk'], 'data/file_index': shard['file_index']},
                          'evidence': [meta_ref(item, 'data/file_index'), row_ref(item['dataset_from_index'], 'index')]})
        frames = item['dataset_to_index']
        kept += 1

    # ---- reasons for every episode that is not kept --------------------------------------------------------
    # Complete encoder outputs left in tmpXXXXXXXX/ ("<key>_<episode:03d>.mp4", dataset_writer.py:81).
    temp_mp4, temp_named = {}, {}
    for temp in temps:
        for item in temp['files']:
            match = re.fullmatch(r'(.+)_([0-9]{3,})\.mp4', Path(item['path']).name)
            if not match or match.group(1) not in video_keys:
                continue
            temp_named[(match.group(1), int(match.group(2)))] = item['path']
            header = item.get('mp4_header') or {}
            if header.get('status') == 'parsed':
                temp_mp4[(match.group(1), int(match.group(2)))] = {'path': item['path'],
                                                                   'samples': header['sample_count']}
    meta_unreadable = sorted(str(p.relative_to(root)) for p in meta_files if p in unreadable)

    def orphan(ep):
        """State of rows that no readable metadata row describes: one whole run? positional index? which camera frames
        that look complete remain for them? Completeness rests on the encoder temp file's name and MP4 header sample
        count and on the staging frame file count; nothing is decoded. The recorder writes unbatched video before
        metadata (dataset_writer.py:326-378) and batched video only after it (dataset_writer.py:380-386)."""
        start, count = first_row[ep], labelled[ep]
        window = slice(start, start + count)
        whole = bool(np.all(lab[window] == ep)) and bool(np.array_equal(frm[window], np.arange(count)))
        reindex = whole and not bool(np.array_equal(idx[window], np.arange(start, start + count)))
        frames = {}
        for key in video_keys:
            temp = temp_mp4.get((key, ep))
            stage = staging.get(ep, {}).get(key, {})
            if temp and temp['samples'] == count:
                frames[key] = {'source': 'encoder_temp_mp4', 'path': temp['path'], 'samples': count}
            elif stage.get('frames') == count:
                frames[key] = {'source': 'staging_frames', 'path': stage['directory'], 'frames': count}
            elif ep < total_episodes and (root / 'videos' / key).is_dir():
                frames[key] = {'source': 'video_files', 'path': str(root / 'videos' / key),
                               'note': 'unbatched video is written before metadata; not verified per episode'}
            elif (key, ep) in temp_named and not work.video_headers:
                frames[key] = {'source': 'encoder_temp_mp4_not_inspected', 'path': temp_named[(key, ep)]}
            else:
                frames[key] = None
        shard = shard_at(start)
        layout = len(np.unique(lab[shard['start']:shard['end']])) == shard['row_groups']
        inspected = all(f is None or f['source'] != 'encoder_temp_mp4_not_inspected' for f in frames.values())
        return {'whole': whole, 'reindex': reindex, 'frames': frames, 'recorder_layout': layout,
                'file': shard['file'], 'first_index': int(idx[start]), 'first_position': start,
                'recoverable': whole and all(frames.values()) and inspected,
                'unknown': whole and all(frames.values()) and not inspected}

    ledger, renumber = {}, {}
    for ep in sorted((set(labelled) | set(claims)) - set(range(kept))):
        positions, reasons, evidence = claims.get(ep, []), [], []
        if not positions:
            if 0 <= ep < total_episodes:
                reason = 'metadata_unreadable_or_unflushed' if meta_unreadable else 'metadata_not_flushed'
            else:
                reason = 'rows_without_metadata'
            ledger[ep] = {'reasons': [reason], 'rows': labelled[ep], 'metadata_rows': 0, 'orphan': orphan(ep),
                          'evidence': [row_ref(first_row[ep], 'episode_index'), row_ref(first_row[ep], 'index')]}
            continue
        if len(positions) > 1:
            reasons.append('duplicate_metadata')
            evidence += [meta_ref(meta_rows[p], 'episode_index') for p in positions[:2]]
        positional = 0 <= ep < len(meta_rows) and meta_rows[ep]['episode_index'] == ep
        item = meta_rows[ep] if positional else meta_rows[positions[0]]
        if not positional:
            reasons.append('metadata_position')
            evidence.append(dict(meta_ref(item, 'episode_index'), expected_row_position=ep))
        problem, extra = span_problem(item, ep)
        if problem:
            reasons.append(problem)
            evidence += extra
        else:
            extra = ambiguous(item, ep)
            if extra:
                reasons.append('ambiguous_rows')
                evidence += extra
        for reason, extra in video.get(ep, []):
            reasons.append(reason)
            evidence += extra
        if not reasons:
            renumber[ep] = {'rows': labelled.get(ep, 0), 'evidence': [meta_ref(item, 'episode_index')]}
            continue
        ledger[ep] = {'reasons': reasons, 'rows': labelled.get(ep, 0), 'metadata_rows': len(positions),
                      'evidence': evidence[:MAX_EVIDENCE]}
    drop = sorted(ledger)
    superseded_meta = sum(len(claims.get(ep, [])) - 1 for ep in range(kept))
    superseded_rows = sum(labelled.get(ep, 0) - meta_rows[ep]['length'] for ep in range(kept))
    missing_committed = [ep for ep in range(kept, total_episodes) if ep not in ledger and ep not in renumber][:MAX_LISTED]

    # In an unresolved ledger, span-derived exclusions are not justified.
    # Keep independent per-episode observations without presenting a drop plan.
    if conflicts:
        independent = {'metadata_without_rows', 'rows_in_unreadable_file', 'metadata_not_flushed',
                       'metadata_unreadable_or_unflushed', 'rows_without_metadata'}
        observed_episodes = []
        for ep in drop:
            reasons = [r for r in ledger[ep]['reasons'] if r in independent]
            if reasons:
                observed_episodes.append({'episode_index': ep, 'reasons': reasons,
                                          'rows': ledger[ep]['rows'], 'metadata_rows': ledger[ep]['metadata_rows']})
        if observed_episodes:
            findings.append({'code': 'salvage.episode_integrity', 'status': 'error',
                'episodes': _ranges(item['episode_index'] for item in observed_episodes),
                'observations': observed_episodes[:MAX_LISTED],
                'message': 'Independent episode defects remain alongside the unresolved ledger: '
                           + '; '.join(f'episode {item["episode_index"]}: {", ".join(item["reasons"])}'
                                       for item in observed_episodes[:3]) + '. No exclusion plan is approved.',
                'remedy': 'Preserve every original file. Restore missing rows or metadata from the source; '
                          'resolve the identity disagreement before deriving a copy plan.',
                'evidence': [e for item in observed_episodes[:2]
                             for e in ledger[item['episode_index']]['evidence'][:2]]})
    if drop and not conflicts:
        visible = [ep for ep in drop if ledger[ep]['metadata_rows']]
        flushed = [ep for ep in drop if ledger[ep]['reasons'][0] in ('metadata_not_flushed',
                                                                    'metadata_unreadable_or_unflushed')]
        unlisted = [ep for ep in drop if ledger[ep]['reasons'] == ['rows_without_metadata']]
        reasons = sorted({r for ep in drop for r in ledger[ep]['reasons']})
        examples = '; '.join(f'episode {ep}: {", ".join(REASON_TEXT[r] for r in ledger[ep]["reasons"][:2])}'
                             for ep in drop[:3])
        parts_text, remedy = [], []
        orphans = flushed + unlisted
        recoverable = [ep for ep in orphans if ledger[ep]['orphan']['recoverable']]
        broken_runs = [ep for ep in orphans if not ledger[ep]['orphan']['whole']]
        unknown_frames = [ep for ep in orphans if ledger[ep]['orphan']['unknown']]
        no_frames = [ep for ep in orphans if ledger[ep]['orphan']['whole'] and not ledger[ep]['orphan']['recoverable']
                     and not ledger[ep]['orphan']['unknown']]
        reindex = [ep for ep in recoverable if ledger[ep]['orphan']['reindex']]
        staged = [ep for ep in recoverable if any(f and f['source'] == 'staging_frames'
                                                  for f in ledger[ep]['orphan']['frames'].values())]
        encoded = [ep for ep in recoverable if any(f and f['source'] == 'encoder_temp_mp4'
                                                   for f in ledger[ep]['orphan']['frames'].values())]
        if flushed:
            parts_text.append(
                f'Episode(s) {_ranges(flushed)[:8]} are committed: info.json counts them (it is rewritten after every saved '
                f'episode, {LOC["totals_per_episode"]}) and their rows sit in files with intact footers, but no readable '
                'metadata row describes them: '
                + (f'their metadata was still buffered ({LOC["metadata_buffered"]}) when the session ended. '
                   if not meta_unreadable else
                   f'their metadata rows were never written ({LOC["metadata_buffered"]}) or sit in the metadata file(s) '
                   f'{meta_unreadable[:4]}, which lost their footer. ')
                + 'LeRobot ignores rows without metadata.')
        if unlisted:
            third_party = sorted({ledger[ep]['orphan']['file'] for ep in unlisted
                                  if not ledger[ep]['orphan']['recorder_layout']})
            parts_text.append(
                f'Episode(s) {_ranges(unlisted)[:8]} have rows that no metadata row describes and that info.json does not '
                f'count. LeRobot ignores rows without metadata: the sampler and len() use only metadata spans and '
                f'total_frames ({LOC["sampler_all_rows"]}; {LOC["len_total_frames"]}). In the pinned recorder this state '
                f'arises when save_episode stops after writing the rows ({LOC["rows_before_metadata"]}); possible causes: '
                f'{CAUSES_SAVE}.'
                + (f' {third_party[:2]} do not follow the recorder\'s one-row-group-per-episode layout, so another tool '
                   'wrote them and these causes may not apply; the bytes record only the state.' if third_party else ''))
        if recoverable:
            def held(key, f):
                if f['source'] == 'encoder_temp_mp4':
                    return f'{key} from encoder temp MP4 {Path(f["path"]).name} ({f["samples"]} samples in its MP4 header)'
                if f['source'] == 'staging_frames':
                    return f'{key} from staging frames {Path(f["path"]).name} ({f["frames"]} frame files)'
                return f'{key} in the video files under videos/{key} (not located per episode)'
            sources = '; '.join(
                f'episode {ep}: ' + ', '.join(held(key, f) for key, f in ledger[ep]['orphan']['frames'].items())
                or f'episode {ep}: rows only' for ep in recoverable[:3])
            used = {f['source'] for ep in recoverable for f in ledger[ep]['orphan']['frames'].values() if f}
            located = [ep for ep in recoverable if any(f and f['source'] == 'video_files'
                                                       for f in ledger[ep]['orphan']['frames'].values())]
            basis = ([] if 'encoder_temp_mp4' not in used else
                     ['for encoder temp MP4s, on the file name and the sample count in the MP4 header'])
            basis += [] if 'staging_frames' not in used else ['for staging frames, on the count of frame files']
            basis += [] if 'video_files' not in used else [
                f'for video files, on info.json counting the episode as saved and the recorder encoding unbatched video '
                f'before it writes metadata ({LOC["video_before_metadata"]}); with no metadata row, where their frames '
                f'sit in those files was not read, and batched encoding writes video only after metadata '
                f'({LOC["batched_video_after"]})']
            parts_text.append(
                f'Episode(s) {_ranges(recoverable)[:8]} are recoverable: their rows form one complete run (frames 0..n-1) '
                + ('and camera frames remain for them (' + sources + '). That rests, ' + '; '.join(basis)
                   + '; no frame was decoded or opened' if video_keys else 'and the dataset has no video feature')
                + '. Regenerating their metadata in a separate copy serves them (observed natively: an episode left by a '
                'SIGINT during concatenation, rows plus its 14-frame encoder temp MP4, served 39 of 39 samples; batched-'
                'encoding episodes from rows plus their staging frames served 54 of 54).')
            remedy.append(
                f'Episode(s) {_ranges(recoverable)[:8]} are recoverable in a separate copy by regenerating their '
                'episode-metadata rows from the data rows'
                + (f'; for {_ranges(encoded)[:8]} place the complete encoder temp MP4 as a video file' if encoded else '')
                + (f'; for {_ranges(staged)[:8]} encode the complete staging frames under images/ first, they are the '
                   'only camera frames' if staged else '')
                + (f'; for {_ranges(located)[:8]} also find where their frames sit in the video files and record that '
                   'video file and timestamps, which no readable metadata row gives' if located else '')
                + (f'; {_ranges(reindex)[:8]} also need their rows re-indexed, because their global index is not their '
                   f'row position (e.g. episode {reindex[0]}: index {ledger[reindex[0]]["orphan"]["first_index"]} at row '
                   f'position {ledger[reindex[0]]["orphan"]["first_position"]}), so metadata alone does not recover them'
                   if reindex else '')
                + '. This metadata-only plan does not do that; keep images/ and tmp* directories with the original.'
                + (' Verify the frames decode (and show the expected episode) before relying on this: no frame was '
                   'decoded here, and the basis is only what the message states.' if video_keys else ''))
        if unknown_frames:
            parts_text.append(f'Episode(s) {_ranges(unknown_frames)[:8]} have one complete run of rows and an encoder temp '
                              'MP4 named for them, but MP4 headers were not read (video_headers=False), so whether it holds '
                              'their complete frames, and so whether they are recoverable, is unknown.')
            remedy.append(f'Episode(s) {_ranges(unknown_frames)[:8]}: rerun with MP4 header reads enabled before deciding; '
                          'keep images/ and tmp* directories with the original until then.')
        if no_frames:
            parts_text.append(f'Episode(s) {_ranges(no_frames)[:8]} have one complete run of rows, but no complete encoder '
                              'temp MP4 or staging frames for them were found here'
                              + (' (MP4 headers were not read, so only staging frame counts were checked)'
                                 if not work.video_headers else '')
                              + ', and with no metadata row describing them, whether a video file holds their frames '
                              'cannot be read from the bytes checked.')
            remedy.append(f'Episode(s) {_ranges(no_frames)[:8]} are recoverable only if their camera frames are found (in '
                          'the video files, a full copy of the repository, or the source recording); then regenerate '
                          'their metadata in a separate copy.')
        if broken_runs:
            parts_text.append(f'Episode(s) {_ranges(broken_runs)[:8]} do not form one complete run of rows.')
            remedy.append(f'Episode(s) {_ranges(broken_runs)[:8]} cannot be rebuilt from their rows alone.')
        if not meta_rows and orphans:
            parts_text.append('No readable episode-metadata row exists at all, so none of these rows can be served '
                              'without regenerating metadata (see the metadata finding).')
        if visible:
            parts_text.append(
                f'{len(visible)} of them have episode-metadata rows that {LEROBOT} uses as-is: it looks metadata up by row '
                f'position ({LOC["positional"]}) and samples every metadata span ({LOC["sampler_all_rows"]}), so as-is it '
                'serves rows under the wrong span or episode, or fails when it decodes their video. Possible causes: an '
                'interrupted or repeated session, a merge of an interrupted dataset, a partial copy, or a converter '
                'defect; the bytes do not say which.')
            remedy.append(f'Episode(s) {_ranges(visible)[:8]} cannot be kept metadata-only; recover them from the source '
                          'recording if it exists.')
        findings.append({'code': 'salvage.episodes_dropped', 'status': 'error' if visible else 'warning',
            'episodes': _ranges(drop), 'reasons': reasons,
            'detected_by': sorted({c for r in reasons for c in EXISTING.get(r, [])}),
            'message': f'{len(drop)} episode(s) cannot be kept by a metadata-only salvage ({examples}). ' + ' '.join(parts_text),
            'remedy': ' '.join(remedy) + ' Apply the plan to a separate copy; never edit the original directory.',
            'evidence': [e for ep in drop[:2] for e in ledger[ep]['evidence'][:2]][:MAX_EVIDENCE]})
    if renumber and not conflicts:
        findings.append({'code': 'salvage.episodes_need_renumbering', 'status': 'warning',
            'episodes': _ranges(renumber),
            'message': f'{len(renumber)} complete episode(s) ({_ranges(renumber)[:8]}) follow an episode the plan cannot '
                       f'keep. Their metadata, spans and rows agree, and LeRobot serves them correctly in the original. A '
                       f'metadata-only copy cannot keep them because LeRobot looks metadata up by row position '
                       f'({LOC["positional"]}): metadata row i must describe episode i, so keeping them means renumbering '
                       'episode_index and index in rewritten data and metadata. They are not lost.',
            'remedy': 'Keep the original for these episodes, or renumber them in a rewritten copy (outside this '
                      'metadata-only plan).',
            'evidence': [e for ep in sorted(renumber)[:2] for e in renumber[ep]['evidence']]})

    samples = work.video_samples(video_keys) if total_episodes > 0 and not meta_rows else {}
    if total_episodes > 0 and not meta_files:
        findings.append({'code': 'salvage.episode_metadata_missing', 'status': 'error',
            'committed_according_to_info': committed, 'video_header_samples': samples,
            'message': f'meta/info.json records {total_episodes} committed episode(s) / {total_frames} frame(s), but '
                       f'meta/episodes/ holds no file. {LEROBOT} buffers episode metadata in memory and writes it only every '
                       f'10 episodes or at finalize ({LOC["metadata_buffered"]}), while info.json is rewritten after every '
                       f'episode ({LOC["totals_per_episode"]}). With the metadata missing, LeRobot falls back to downloading '
                       f'the repository from the Hub ({LOC["hub_fallback"]}), so its error names a network or repository '
                       f'problem rather than the missing files; --resume fails the same way. Possible causes: {CAUSES_HARD}; '
                       'or finalize() raised before flushing (observed natively with batched video encoding).',
            'remedy': 'info.json is the only surviving record of how many episodes were committed; their spans, video '
                      'offsets and per-episode statistics were never written, so a metadata-only plan cannot restore them. '
                      'Keep the original bytes. Rows in files with intact footers can be recovered by regenerating metadata '
                      'from them in a separate copy (see salvage.episodes_dropped); where no video was written, the '
                      'complete staging frames under images/ are the only camera frames and must be encoded first '
                      '(see salvage.leftovers). Otherwise re-record.',
            'evidence': declared})

    if staging or temps:
        never_saved = sorted(ep for ep in staging if ep >= total_episodes and not labelled.get(ep))
        uninspected = not work.video_headers and any(Path(f['path']).suffix == '.mp4' for t in temps for f in t['files'])
        useful = {ep: [(key, f) for key, f in ledger[ep]['orphan']['frames'].items()
                       if f and f['source'] in ('staging_frames', 'encoder_temp_mp4')]
                  for ep in ledger if 'orphan' in ledger[ep]}
        useful = {ep: items for ep, items in useful.items() if items}
        only = sorted(ep for ep, items in useful.items()
                      if any(f['source'] == 'staging_frames' and not (root / 'videos' / key).is_dir() for key, f in items))
        held = ('; '.join(f'episode {ep}: ' + ', '.join(f'{f["source"].replace("_", " ")} {Path(f["path"]).name} '
                                                        f'({f.get("samples", f.get("frames"))} frames, as many as its rows)'
                                                        for _, f in items)
                          for ep, items in sorted(useful.items())[:3]))
        evidence = [{'path': value['directory'], 'sha256': None, 'pointer': key,
                     'value': {'episode_index': ep, 'frames': value['frames']}}
                    for ep in sorted(staging)[:2] for key, value in staging[ep].items()]
        evidence += [{'path': f['path'], 'sha256': f.get('sha256'), 'pointer': 'encoder temp file',
                      'value': {k: f[k] for k in ('size_bytes', 'mp4_header') if k in f}}
                     for t in temps[:2] for f in t['files'][:1]]
        findings.append({'code': 'salvage.leftovers', 'status': 'warning',
            'staging_episodes': sorted(staging)[:MAX_LISTED], 'encoder_temp_dirs': [t['directory'] for t in temps],
            'message': (f'Staging frames remain under images/ for episode(s) {sorted(staging)[:8]}. ' if staging else '')
                       + (f'{len(temps)} encoder temp director(ies) remain in the dataset root. ' if temps else '')
                       + f'{LEROBOT} writes camera frames to images/ until save_episode encodes them ({LOC["staging"]}) '
                       f'and encodes each episode in a tmpXXXXXXXX directory inside the dataset root ({LOC["temp_dirs"]}); '
                       'both are removed on a normal finish and LeRobot never loads them.'
                       + (f' Episode(s) {never_saved[:8]} are at or beyond info.total_episodes={total_episodes} and have no '
                          'rows: their numeric rows were held in memory until save_episode and are lost.'
                          if never_saved else '')
                       + (f' They hold camera frames that look complete (by encoder file name and MP4 header sample count, '
                          f'or staging file count; not decoded) for episode(s) {_ranges(useful)[:8]} whose rows remain '
                          f'({held}).' if useful else '')
                       + (f' For episode(s) {only[:8]} no videos/ directory exists, so these staging frames are the only '
                          'camera frames.' if only else '')
                       + (f' Frames left for episode {total_episodes}, the next index --resume records, are encoded into '
                          f'that resumed episode\'s video because the encoder takes every frame-NNNNNN file in the '
                          f'directory ({LOC["staging_glob"]}); observed natively, the file gained the stale frames and the '
                          'episode\'s video interval grew, while the frames LeRobot served were the new ones.'
                          if staging.get(total_episodes) else '')
                       + (' Encoder temp MP4s were not inspected (video_headers=False), so what they hold is unknown.'
                          if uninspected else '')
                       + ' Possible causes: an interruption during recording or during video encoding.',
            'remedy': (f'Keep images/ and the tmp* directories with the original: they hold the camera frames needed to '
                       f'recover episode(s) {_ranges(useful)[:8]}. Recovering them means encoding the staging frames, or '
                       'placing the complete encoder temp MP4, as a video file and regenerating metadata in a separate '
                       'copy (observed natively: 54 of 54 and 39 of 39 samples served). Completeness here rests on file '
                       'names, MP4 header sample counts and staging file counts; verify the frames decode before relying '
                       'on it. A metadata-only copy leaves these directories out.'
                       + (' Encoder temp MP4s were not inspected (video_headers=False); rerun with header reads to judge '
                          'them.' if uninspected else '') if useful else
                       'Keep images/ and the tmp* directories with the original: their encoder temp MP4s were not '
                       'inspected (video_headers=False), so whether they hold complete frames for episodes whose rows '
                       'remain is unknown until their headers are read. A metadata-only copy leaves these directories '
                       'out.' if uninspected else
                       'Leave images/ and tmp* directories out of a metadata-only copy (LeRobot never loads them) and keep '
                       'them with the original; no complete frames for an episode whose rows remain were found in them '
                       '(judged by file name, MP4 header sample count and staging file count).'),
            'evidence': evidence[:MAX_EVIDENCE]})

    if total_episodes == 0 and not meta_rows and not rows and not unreadable:
        findings.append({'code': 'salvage.nothing_committed', 'status': 'warning', 'detected_by': ['episode.empty_dataset'],
            'message': 'meta/info.json records no committed episode and no metadata or data rows exist; there is '
                       'nothing to salvage.',
            'remedy': 'Re-record. Frames of a session interrupted before its first save_episode were never written as rows.',
            'evidence': declared})

    # ---- task table and statistics (both read at load) ---------------------------------------------------
    has_rows = bool(rows or meta_rows)
    tasks_missing = has_rows and not tasks.exists()
    tasks_fatal_zero = has_rows and task_count is not None and task_count > 0 and total_tasks == 0
    tasks_short = bool(rows) and task_count is not None and max_task is not None and max_task >= task_count
    if tasks_short:
        findings.append({'code': 'salvage.tasks_unusable', 'status': 'error',
            'message': f'meta/tasks.parquet holds {task_count} task(s) while data rows use task_index up to {max_task}. '
                       f'{LEROBOT} looks each sample\'s task up by its position in that table ({LOC["task_lookup"]}), so '
                       'every sample whose task_index lies past the table fails (observed natively for an empty task '
                       'table: IndexError on all 87 samples). Possible causes: the task table was rewritten or truncated '
                       'after the rows were recorded, or a copy left it incomplete.',
            'remedy': 'Restore meta/tasks.parquet from the source or a backup; task strings cannot be recovered from rows, '
                      'which hold only task_index. The plan leaves the table out until it is restored.',
            'evidence': [_evidence(work.artifact(tasks, 'dataset_tasks'), '/num_rows', task_count)]})
    if tasks_missing or tasks_fatal_zero:
        findings.append({'code': 'salvage.tasks_unusable', 'status': 'error',
            'message': (f'meta/tasks.parquet is absent while rows exist. {LEROBOT} reads it when info.total_tasks > 0 '
                        f'({LOC["skip_when_zero"]}) and then fails into the Hub fallback ({LOC["hub_fallback"]}); with '
                        'total_tasks 0 it loads no task table and every served sample fails at the task lookup '
                        f'({LOC["task_lookup"]}).' if tasks_missing else
                        f'info.json declares total_tasks=0 while meta/tasks.parquet holds {task_count} task(s) and rows '
                        f'exist. {LEROBOT} loads the task table only when total_tasks > 0 ({LOC["skip_when_zero"]}), so '
                        f'every served sample fails at the task lookup ({LOC["task_lookup"]}; observed natively: '
                        'AttributeError on every sample).')
                       + ' Possible causes: a session interrupted around its first save, a tool that reset info.json, or a '
                       'copy that left the task table behind.',
            'remedy': ('Restore meta/tasks.parquet from the source; task strings cannot be recovered from rows, which hold '
                       'only task_index.' if tasks_missing else
                       f'Set total_tasks={task_count} in the salvage copy (listed in the plan).'),
            'evidence': [_evidence(info_art, '/total_tasks', total_tasks)]})
    stats_missing = has_rows and stats_art is None
    if stats_missing:
        findings.append({'code': 'salvage.stats_missing', 'status': 'warning',
            'message': f'meta/stats.json is absent. {LEROBOT} loads without it ({LOC["load_stats"]} returns None), but '
                       'normalization statistics for training then have to be recomputed from the rows.',
            'remedy': 'Recompute statistics from the kept rows before training.', 'evidence': []})

    if unlocatable:
        findings.append({'code': 'salvage.feature_unlocatable', 'status': 'error', 'video_keys': unlocatable,
            'detected_by': ['video.feature_without_index'],
            'message': f'info.json declares video feature(s) {unlocatable}, but no episode-metadata column locates them '
                       f'for any episode. {LEROBOT} reads videos/<key>/chunk_index, file_index and from_timestamp for every '
                       'frame it serves (src/lerobot/datasets/dataset_metadata.py:356-358, '
                       'src/lerobot/datasets/dataset_reader.py:434-437), so no episode loads while the feature is declared. '
                       'This is a feature-level defect, not a per-episode one, so it does not decide which episodes are '
                       'complete. Possible causes: the camera failed or was disconnected for the whole session, a '
                       'converter dropped the columns, or the feature was declared after recording.',
            'remedy': 'Restore the location columns from the source recording. Otherwise, in the salvage copy only, '
                      'remove the feature from info.json features, accepting the loss of that camera; that choice belongs '
                      'to the dataset owner.',
            'evidence': [_evidence(info_art, f'/features/{key}/dtype', 'video') for key in unlocatable[:MAX_EVIDENCE]]})
    if truncated:
        affected = sorted({ep for item in truncated.values() for ep in item['episodes']})
        findings.append({'code': 'salvage.video_file_truncated', 'status': 'error',
            'files': sorted(truncated)[:MAX_LISTED], 'episodes': _ranges(affected),
            'message': f'{len(truncated)} referenced video file(s) are shorter than their own MP4 boxes declare (e.g. '
                       f'{sorted(truncated)[0]}: {next(iter(truncated.values()))["reason"]}). Frames stored past the cut '
                       'cannot be decoded, and LeRobot fails when a sample needs them. Which of the referencing episodes '
                       f'({_ranges(affected)[:8]}) still decode cannot be decided from headers, so their video is '
                       f'unverified. Possible causes: an interrupted cross-filesystem move after concatenation '
                       f'({LOC["concat_move"]}), a truncated copy or upload, or a full disk.',
            'remedy': 'Restore the video file from the source or a backup. The plan does not verify video for these '
                      'episodes.',
            'evidence': [{'path': str(root / rel), 'sha256': item['sha256'], 'pointer': 'mp4 top-level boxes',
                          'value': item['reason']} for rel, item in sorted(truncated.items())[:MAX_EVIDENCE]]})

    rebuild = []
    if stats_problem:
        rebuild.append({'file': 'meta/stats.json', 'how': 'recompute statistics from the kept rows, or restore the file; '
                        'LeRobot fails to load while it is invalid JSON'})
    if tasks in unreadable or tasks_missing or tasks_short:
        rebuild.append({'file': 'meta/tasks.parquet', 'how': 'restore it from the source or a backup; task strings cannot '
                        'be recovered from the rows'})
    video_verified = not video_keys or (work.coverage.get('video_index_status') in ('complete', 'not_applicable')
                                        and not truncated)
    plan = {'strategy': 'unknown', 'validity': {}} if conflicts else _plan(
                 work, info, total_episodes, total_frames, total_tasks, None if tasks_short else task_count,
                 meta_rows, kept, frames, fixes,
                 spans, unreadable, meta_files, lab, video_keys, template, chunks_size, staging, temps, rebuild,
                 video_verified, tasks_fatal_zero, repaired_spans)
    if plan.get('validity') is not None:
        plan['validity']['blocked_until_features_restored_or_removed'] = unlocatable
        plan['validity']['video_verified'] = video_verified
        plan['validity']['episodes_with_unverified_video'] = _ranges(
            ep for item in truncated.values() for ep in item['episodes'] if ep < kept)
    result['plan'] = plan
    clean_row_ledger = (bool(meta_rows) and kept == len(meta_rows) and not drop and not unreadable
                        and not fixes and not superseded_meta and not superseded_rows
                        and rows == frames == meta_rows[-1]['dataset_to_index'])
    result['assessment_kind'] = ('metadata_rows_mismatch' if repaired_spans else
                                 'metadata_totals_mismatch' if clean_row_ledger and plan.get('changes', {}).get('totals')
                                 else 'recording_recovery' if drop or unreadable or staging or temps
                                 else 'recording_integrity')
    if not conflicts and plan['strategy'] in ('metadata_copy', 'keep_as_is', 'rebuild_required'):
        changes = plan['changes']
        reads_differently = bool(changes['totals'] or changes['metadata_rows_removed']
                                or tasks_fatal_zero or repaired_spans)
        text = [f'A metadata-only salvage keeps episodes 0-{kept - 1} ({frames} frames)']
        if changes['totals']:
            text.append(f'sets {plan["files"]["rewrite"][0]["set"]} in info.json (declared '
                        f'{total_episodes} episodes / {total_frames} frames / {total_tasks} tasks; LeRobot reads these '
                        f'directly, {LOC["len_total_frames"]}; {LOC["skip_when_zero"]})')
        if total_episodes == 0 and meta_rows:
            text.append(f'(as-is LeRobot skips the episode metadata because total_episodes is 0, {LOC["skip_when_zero"]}, '
                        'and serves an empty dataset)')
        if changes['metadata_rows_removed']:
            text.append(f'removes {changes["metadata_rows_removed"]} metadata row(s) that LeRobot samples as-is')
        if fixes:
            text.append(f'corrects {len(fixes)} data-file pointer(s)')
        if repaired_spans:
            text.append(f'rebuilds {len(repaired_spans)} row span(s) from intact episode rows')
        if plan['files']['omit_count']:
            text.append(f'omits {plan["files"]["omit_count"]} file(s)')
        if superseded_meta or superseded_rows:
            text.append(f'and excludes {superseded_meta} metadata row(s) and {superseded_rows} data row(s) that also claim '
                        'kept episode indices')
        if any(r['file'] == 'meta/tasks.parquet' for r in rebuild):
            text.append('but the copy loads only after meta/tasks.parquet is restored')
        if any(r['file'] == 'meta/stats.json' for r in rebuild):
            text.append('and leaves meta/stats.json out: the copy loads without it, but statistics must be recomputed '
                        'from the kept rows before training or resuming')
        if unlocatable:
            text.append(f'but the kept episodes load only after camera(s) {unlocatable} are restored or removed '
                        '(salvage.feature_unlocatable)')
        findings.append({'code': 'salvage.metadata_plan',
            'status': 'error' if (reads_differently or rebuild) else 'warning',
            'keep': _ranges(range(kept)), 'drop': _ranges(drop),
            'message': ', '.join(text) + '. ' + (
                'LeRobot reads the original differently from this plan.' if reads_differently else
                'Loading the original fails until the listed files are rebuilt.' if rebuild else
                'LeRobot already reads the kept episodes this way; the plan records what is excluded and why.')
                       + ('' if video_verified else ' The plan verifies rows and metadata only; video was not verified.'),
            'remedy': 'Apply the plan to a new directory: copy the listed files unchanged, write the listed info.json fields '
                      'and metadata rows, rebuild any listed file, then rerun this check and a native loader smoke test on '
                      'the copy.',
            'evidence': declared})
    elif not conflicts and plan['strategy'] == 'not_metadata_only' and total_episodes > 0 and meta_files:
        findings.append({'code': 'salvage.not_metadata_only', 'status': 'error',
            'committed_according_to_info': committed, 'video_header_samples': samples,
            'message': f'No episode can be kept by a metadata-only salvage: {plan["reason"]}. meta/info.json records '
                       f'{total_episodes} committed episode(s) / {total_frames} frame(s).',
            'remedy': 'Keep the original bytes. Recovery needs byte-level work on a separate copy, or re-recording.',
            'evidence': declared})

    refused = (bool(broken) or tasks_missing
               or any(not str(p.relative_to(root)).startswith('data/') for p in unreadable))
    hazard = _resume_hazard(work, info_art, total_episodes, total_frames, original_meta_rows, rows,
                            [p for p in unreadable if str(p.relative_to(root)).startswith('data/')], video_keys,
                            template, chunks_size, refused)
    if hazard:
        findings.append(hazard)
    if isinstance(plan.get('validity'), dict):
        staged_next_camera_frames = any(
            key in video_keys and item.get('frames', 0) > 0
            for key, item in staging.get(total_episodes, {}).items())
        plan['validity']['current_directory_resume_status'] = (
            'unsafe' if hazard or staged_next_camera_frames else
            'unknown' if refused or not meta_rows else 'no_hazard_detected')
        plan['validity']['current_directory_resume_hazard_codes'] = (
            (['salvage.resume_hazard'] if hazard else []) +
            (['salvage.leftovers'] if staged_next_camera_frames else []))
    video_state = work.coverage.get('video_index_status')
    if video_keys and (kept or conflicts) and video_state not in ('complete', 'not_applicable'):
        reason = work.coverage.get('video_index') or ''
        if truncated:
            reason = (f'{len(truncated)} file(s) truncated, see salvage.video_file_truncated; '
                      + reason.replace('with unsupported MP4 layout not checked', 'not parseable from headers'))
        findings.append({'code': 'salvage.video_unchecked', 'status': 'unknown',
            'message': f'Video completeness of the kept episodes was not fully checked ({reason}). Rows and metadata were; '
                       'the plan does not cover video.',
            'remedy': 'Make the video files available locally with MP4 header reads enabled, then rerun.', 'evidence': []})
    result['episodes'] = {
        'keep': _ranges(range(kept)),
        'drop': [{'episode_index': ep, 'reasons': ledger[ep]['reasons'], 'rows': ledger[ep]['rows'],
                  'metadata_rows': ledger[ep]['metadata_rows'], 'evidence': ledger[ep]['evidence'][:2]}
                 for ep in drop[:MAX_LISTED]],
        'drop_truncated': len(drop) > MAX_LISTED,
        'excluded_from_plan_meaning': 'Excluded from this metadata-only copy; retain original bytes. Not an instruction to delete episodes.',
        'recovery_candidates': _ranges(ep for ep in drop if ledger[ep].get('orphan', {}).get('recoverable')),
        'recovery_candidate_scope': 'Complete row runs and surviving frame evidence; actual frame decoding and regenerated metadata remain unverified.',
        'need_renumbering': _ranges(renumber),
        'committed_but_not_verifiable': _ranges(missing_committed),
        'staging_only': sorted(ep for ep in staging if ep not in ledger and ep >= kept)[:MAX_LISTED],
        'superseded_claims_on_kept': {'metadata_rows': superseded_meta, 'data_rows': superseded_rows},
        'committed_according_to_info': committed}
    result['counts'].update(episodes_seen=len(ledger) + len(renumber) + kept, keep=kept, drop=len(drop),
                            need_renumbering=len(renumber), staging_episodes=len(staging), encoder_temp_dirs=len(temps),
                            files_without_footer=len(unreadable))
    work.coverage['complete'] = not unreadable and video_verified
    if conflicts:
        # Finish every independent check, including resume on ORIGINAL metadata,
        # before withholding the operation. Verified prefix evidence is useful
        # but is not permission to copy an unresolved remainder away.
        validity = plan.get('validity') or {}
        current = ('unsafe' if (hazard or conflicts['physical_span_conflicts']
                               or conflicts['metadata_identity_conflicts']) else
                   validity.get('current_directory_resume_status', 'unknown'))
        hazard_codes = validity.get('current_directory_resume_hazard_codes', [])
        if conflicts['physical_span_conflicts']:
            hazard_codes = [*hazard_codes, 'salvage.metadata_row_conflict']
        if conflicts['metadata_identity_conflicts']:
            hazard_codes = [*hazard_codes, 'salvage.metadata_row_conflict']
        result['assessment_kind'] = 'metadata_rows_unresolved'
        result['plan'] = {'strategy': 'unknown', 'reason': 'Row/metadata contradictions require review.',
            'repair_blockers': blockers,
            'validity': {'resume_safe': False,
                         'current_directory_resume_status': current if current == 'unsafe' else 'unknown',
                         'current_directory_resume_hazard_codes': hazard_codes,
                         'resume_blockers': ['Resolve row/metadata contradictions before any copy or resume.']}}
        unresolved = (set(labelled) | set(claims) | set(range(total_episodes))) - set(range(kept))
        result['episodes'] = {'keep': _ranges(range(kept)), 'drop': [], 'unresolved': _ranges(unresolved),
            'keep_scope': 'Verified physical prefix only; no copy or exclusion plan is approved.',
            'excluded_from_plan_meaning': 'No exclusion plan exists. Preserve every original episode and file.'}
        result['counts'].update(episodes_seen=kept + len(unresolved), keep=kept, drop=0,
                               unresolved=len(unresolved), need_renumbering=0)
        work.coverage['complete'] = False


def _video_reasons(work, meta_rows, meta_files, unreadable, video_keys, template):
    """Per-episode video drop reasons, feature-level unlocatable cameras and truncated video files, from the existing
    video-index check, the location columns already read and MP4 top-level boxes; no second detector."""
    if not meta_rows or any(p in unreadable for p in meta_files):
        work.coverage.update(video_index='not checked: episode metadata unavailable',
                             video_index_status='unknown' if meta_rows else None)
        return {}, [], {}
    from .video_index import check_video_index
    try:
        found, coverage = check_video_index(work.root, max_rows=min(work.max_rows, 200000),
                                            max_files=min(work.max_files, 1024), video_headers=work.video_headers)
    except (OSError, ValueError) as exc:
        # If the existing index check stops (for example on one of its own budgets), keep going with this module's
        # own containment-checked per-episode checks instead of skipping them; video stays unverified.
        found, coverage = [], {'video_index': f'not checked: {exc}'[:200], 'video_index_details': {'status': 'unknown'}}
    details = coverage.get('video_index_details', {})
    work.coverage.update(video_index=coverage.get('video_index'), video_index_status=details.get('status'))
    out, unlocatable = {}, []
    for finding in found:
        if finding['code'] == 'video.interval_past_end':
            listed = finding.get('episodes', [])
            for item in listed:
                out.setdefault(item['episode_index'], []).append(('video_interval_past_end', [{
                    'path': str(work.root / item['file']), 'sha256': None, 'pointer': item['video_key'],
                    'value': {'file_samples': item['file_samples'], 'episode_last_frame_s': item['episode_last_frame_s']}}]))
            if len(listed) >= 32:  # the existing check lists at most 32 overruns
                work.coverage['video_index_status'] = 'partial'
        elif finding['code'] == 'video.feature_without_index' and not finding.get('episodes'):
            unlocatable.append(finding.get('video_key'))  # whole location columns are absent: a feature-level defect
    referenced = {}
    for item in meta_rows:
        for key in video_keys:
            if key in unlocatable or f'videos/{key}/chunk_index' not in item:
                continue
            chunk, file = item.get(f'videos/{key}/chunk_index'), item.get(f'videos/{key}/file_index')
            if chunk is None or file is None:
                out.setdefault(item['episode_index'], []).append(
                    ('video_unindexed', [_evidence(item['_artifact'], f'row {item["_row"]}/videos/{key}/chunk_index',
                                                   chunk)]))
                continue
            if not _integer(chunk) or not _integer(file):
                continue
            rel = _video_rel(template, key, chunk, file)
            referenced.setdefault(rel, set()).add(item['episode_index'])
            camera = work.root / 'videos' / key
            if camera.is_dir() and not (work.root / rel).exists():
                out.setdefault(item['episode_index'], []).append(
                    ('video_file_missing', [{'path': str(work.root / rel), 'sha256': None, 'pointer': key,
                                             'value': 'absent'}]))
    truncated = {}
    if work.video_headers:
        for rel in sorted(referenced)[:MAX_VIDEO_HEADERS]:
            path = work.path(rel)
            if not path.is_file():
                continue
            timing = work.mp4(path)
            if timing.get('status') == 'unsupported' and 'escapes its file' in str(timing.get('reason')):
                truncated[rel] = {'reason': f'a top-level MP4 box extends past the {path.stat().st_size}-byte file end',
                                  'sha256': work.artifact(path, 'video_truncated')['sha256'],
                                  'episodes': sorted(referenced[rel])}
    return out, sorted(k for k in unlocatable if isinstance(k, str)), truncated


def _plan(work, info, total_episodes, total_frames, total_tasks, task_count, meta_rows, kept, frames, fixes, spans,
          unreadable, meta_files, lab, video_keys, template, chunks_size, staging, temps, rebuild, video_verified,
          tasks_fatal_zero, span_fixes=()):
    import numpy as np
    root = work.root
    if not meta_rows or kept == 0:
        reason = ('no readable episode-metadata row exists (' + ('metadata was never flushed' if not meta_files else
                  'the metadata files have no valid footer') + ')') if not meta_rows else \
            'episode 0 is not complete, and LeRobot needs metadata row i to describe episode i'
        return {'strategy': 'not_metadata_only', 'reason': reason,
                'committed_according_to_info': {'total_episodes': total_episodes, 'total_frames': total_frames}}
    removed = len(meta_rows) - kept
    corrected = {'total_episodes': kept, 'total_frames': frames}
    if task_count is not None:
        corrected['total_tasks'] = task_count
    if kept != total_episodes:
        corrected['splits'] = {'train': f'0:{kept}'}
    info_set = {k: v for k, v in corrected.items() if info.get(k) != v}
    if 'total_tasks' in info_set and not (tasks_fatal_zero or set(info_set) - {'total_tasks'}):
        info_set.pop('total_tasks')  # a nonzero mismatch does not change loading; only rewrite it with other fields
    totals = bool(set(info_set) & {'total_episodes', 'total_frames', 'total_tasks'})
    copied = [s for s in spans if s['start'] < frames]
    leftover = sum(max(0, s['end'] - frames) for s in copied)
    tail_labels = set()
    for shard in copied:
        if shard['end'] > frames:
            tail_labels |= {int(v) for v in np.unique(lab[max(frames, shard['start']):shard['end']])}
    positions = {}
    for position, item in enumerate(meta_rows):
        positions.setdefault(item['_file'], []).append(position)
    fixed = {f['episode_index']: f['to'] for f in fixes}
    changed_spans = {f['episode_index'] for f in span_fixes}
    kept_meta = sorted(name for name, rows_at in positions.items() if min(rows_at) < kept)
    pure_meta = [name for name in kept_meta if max(positions[name]) < kept
                 and not (set(fixed) | changed_spans) & set(positions[name])]
    rewrite = [{'file': 'meta/info.json', 'set': info_set}] if info_set else []
    for name in kept_meta:
        if name not in pure_meta:
            inside = [p for p in positions[name] if p < kept]
            rewrite.append({'file': name, 'keep_physical_rows': [meta_rows[inside[0]]['_row'], meta_rows[inside[-1]]['_row']],
                            'drop_physical_rows': len(positions[name]) - len(inside),
                            'pointer_corrections': len(set(fixed) & set(inside))})
            if changed_spans & set(inside):
                rewrite[-1]['row_span_corrections'] = len(changed_spans & set(inside))
    omit = ({s['file'] for s in spans if s['start'] >= frames}
            | ({str(p.relative_to(root)) for p in meta_files} - set(kept_meta))
            | {str(p.relative_to(root)) for p in unreadable})
    videos = set()
    for i in range(kept):
        for key in video_keys:
            chunk, file = meta_rows[i].get(f'videos/{key}/chunk_index'), meta_rows[i].get(f'videos/{key}/file_index')
            if _integer(chunk) and _integer(file):
                videos.add(_video_rel(template, key, chunk, file))
    # This list describes a fresh directory assembled from the plan, not a
    # deletion recipe for a recursive copy. Still account for every observed
    # video-tree file so an orphan video cannot disappear behind lists_complete.
    omit |= set(work.video_inventory()) - videos
    unchanged = [s['file'] for s in copied] + pure_meta + sorted(videos)
    if not info_set:
        unchanged.append('meta/info.json')  # always part of the copy (LeRobot cannot load without it)
    for name in ('meta/tasks.parquet', 'meta/stats.json'):
        if (root / name).exists() and root / name not in unreadable and name not in {r['file'] for r in rebuild}:
            unchanged.append(name)
    for relative in unchanged + sorted(omit) + [r['file'] for r in rewrite]:
        _safe_relative(relative)
    for relative in unchanged + [r['file'] for r in rewrite]:
        if os.path.lexists(root / relative):  # invariant: a plan copies or rewrites only regular files, never a link
            work.intake.regular(root / relative)
    last = meta_rows[kept - 1]
    data_pointer = fixed.get(kept - 1, {'data/chunk_index': last['data/chunk_index'],
                                        'data/file_index': last['data/file_index']})
    targets = ['data/chunk-{:03d}/file-{:03d}.parquet'.format(*_next(data_pointer['data/chunk_index'],
                                                                      data_pointer['data/file_index'], chunks_size))]
    if _integer(last.get('meta/episodes/chunk_index')) and _integer(last.get('meta/episodes/file_index')):
        targets.append('meta/episodes/chunk-{:03d}/file-{:03d}.parquet'.format(
            *_next(last['meta/episodes/chunk_index'], last['meta/episodes/file_index'], chunks_size)))
    for key in video_keys:
        chunk, file = last.get(f'videos/{key}/chunk_index'), last.get(f'videos/{key}/file_index')
        if _integer(chunk) and _integer(file):
            targets.append(_video_rel(template, key, *_next(chunk, file, chunks_size)))
    blockers = []
    if leftover:
        blockers.append(f'{leftover} non-kept row(s) remain after the kept rows in copied data file(s); resumed episodes '
                        'would be stored after them while their spans start at the kept total')
    written = set(unchanged) | {r['file'] for r in rewrite}
    for target in targets:
        if target in written:
            blockers.append(f'resume would replace copied {target}')
    rebuilt = {r['file'] for r in rebuild}
    if 'meta/tasks.parquet' in rebuilt:
        blockers.append('the copy does not load until meta/tasks.parquet is restored')
    if 'meta/stats.json' in rebuilt:
        blockers.append('the copy loads without meta/stats.json, but a resume would write it from the new episodes only '
                        '(src/lerobot/datasets/dataset_metadata.py:682-683) until statistics are rebuilt')
    data_rewrite = [
        {'file': shard['file'], 'keep_physical_rows': [0, frames - shard['start']],
         'range_end': 'exclusive', 'rows_to_exclude': shard['end'] - frames,
         'how': 'On a separate candidate, keep only these physical rows and preserve the Arrow schema metadata; '
                'this data rewrite is outside the metadata-only plan. Rerun this check and the native loader before resume.'}
        for shard in copied if shard['end'] > frames
    ]
    changed = info_set or removed or fixes or span_fixes or omit
    strategy = ('rebuild_required' if rebuild else 'metadata_copy' if changed else
                'keep_as_is' if leftover else 'none_needed' if video_verified else 'none_needed_rows_only')
    return {
        'strategy': strategy,
        'keep': _ranges(range(kept)), 'kept_episodes': kept, 'kept_frames': frames,
        'corrected_info': corrected,
        'declared_info': {'total_episodes': total_episodes, 'total_frames': total_frames, 'total_tasks': total_tasks,
                          'splits': info.get('splits')},
        'metadata_rows': {'keep_row_positions': [0, kept - 1], 'source_files': kept_meta[:MAX_LISTED],
                          'note': 'Kept rows stay verbatim apart from listed pointer and row-span corrections. A rewritten metadata file '
                                  'keeps its name, so each row\'s meta/episodes/chunk_index and file_index stay valid; '
                                  'resume continues after the last row\'s pointer '
                                  '(src/lerobot/datasets/dataset_metadata.py:588-601).'},
        'pointer_corrections': [{k: f[k] for k in ('episode_index', 'from', 'to')} for f in fixes[:PLAN_LIST_LIMIT]],
        'pointer_correction_evidence': [e for f in fixes[:2] for e in f['evidence']],
        'row_span_corrections': [{k: f[k] for k in ('episode_index', 'from', 'to')}
                                 for f in span_fixes[:PLAN_LIST_LIMIT]],
        'rebuild': rebuild,
        'required_data_rewrite': data_rewrite,
        'files': {'copy_unchanged': unchanged[:PLAN_LIST_LIMIT], 'copy_unchanged_count': len(unchanged),
                  'rewrite': rewrite[:PLAN_LIST_LIMIT], 'rewrite_count': len(rewrite),
                  'omit': sorted(omit)[:PLAN_LIST_LIMIT], 'omit_count': len(omit),
                  'lists_complete': max(len(unchanged), len(rewrite), len(omit), len(fixes), len(span_fixes)) <= PLAN_LIST_LIMIT,
                  'inventory_scope': 'Known native data/episode metadata plus every bounded local videos/ file; '
                                     'not a whole-directory deletion inventory. Build a fresh directory from the copy/rewrite lists.',
                  'left_out_keep_with_original': ((['images/'] if staging else [])
                                                  + [Path(t['directory']).name + '/' for t in temps])[:16]},
        'changes': {'totals': totals, 'info_fields': sorted(info_set), 'metadata_rows_removed': removed,
                    'pointer_corrections': len(fixes), 'row_span_corrections': len(span_fixes),
                    'rows_left_unreferenced_in_copied_files': leftover},
        'validity': {
            'unfiltered_loading': 'The kept episodes tile rows [0, kept_frames) with index equal to row position, which is '
                                  'how LeRobot reads rows when no episode list is given.',
            'episode_selection_unsafe_for': _ranges(v for v in tail_labels if 0 <= v < kept),
            'resume_safe': not blockers, 'resume_blockers': blockers,
            'resume_safe_applies_to': 'The predicted separate copy after every listed operation, never the current directory.',
            'copy_verification_required': True},
        'stats_note': ('meta/stats.json aggregates every saved episode including dropped ones; recompute it from the kept '
                       'rows if exact normalization statistics matter.' if (removed or totals) else None),
    }


def _resume_hazard(work, info_art, total_episodes, total_frames, meta_rows, rows, unreadable, video_keys, template,
                   chunks_size, refused):
    """What lerobot-record --resume would do to this directory as it is.

    Resume reads info, stats, tasks and every episode-metadata file but no data file. When one of those is missing or
    unreadable it fails loudly (observed natively) and the file finding already says so; footer-less *data* files do
    not stop it (observed natively after a hard kill of a multi-file recording)."""
    if refused or not meta_rows:
        return None
    last = meta_rows[-1]
    reasons, evidence = [], []
    footerless = sorted(str(p.relative_to(work.root)) for p in unreadable)
    if footerless:
        reasons.append(f'{len(footerless)} footer-less data file(s) remain (e.g. {footerless[0]}); resume reads no data '
                       'file, so it proceeds, while loading keeps failing until they are removed')
    if total_episodes == 0:
        # The loader does not read episode metadata at all: resume restarts at episode 0 / index 0 / file-000.
        reasons.append(f'info.json records 0 episodes while {len(meta_rows)} episode-metadata row(s) exist; the loader '
                       f'skips episode metadata when total_episodes is 0 ({LOC["skip_when_zero"]}), so resume starts at '
                       f'episode 0, global index 0 and file-000 and replaces the data, metadata and video file-000 '
                       f'({LOC["resume_from_zero"]}); observed natively, every committed episode stored in those files '
                       'was replaced by the new ones')
        evidence.append(_evidence(info_art, '/total_episodes', total_episodes))
        targets = [('data', 'data/chunk-000/file-000.parquet'), ('metadata', 'meta/episodes/chunk-000/file-000.parquet')]
        targets += [(f'video {key}', _video_rel(template, key, 0, 0)) for key in video_keys]
    else:
        if rows > last['dataset_to_index']:
            reasons.append(f'{rows - last["dataset_to_index"]} readable data row(s) lie after the last metadata span (which '
                           f'ends at {last["dataset_to_index"]}); resumed episodes get spans starting there but are stored '
                           'after those rows, so the reader serves them instead (observed natively: 14 of 20 resumed rows '
                           'never served)')
            evidence.append(_evidence(last['_artifact'], f'row {last["_row"]}/dataset_to_index', last['dataset_to_index']))
        if total_episodes != len(meta_rows) or total_frames != last['dataset_to_index']:
            reasons.append(f'info.json totals ({total_episodes} episodes / {total_frames} frames) differ from the metadata '
                           f'({len(meta_rows)} rows ending at {last["dataset_to_index"]}); resume labels new rows from the '
                           f'totals ({LOC["resume_labels"]}) but places their spans after the last metadata row, so episode '
                           'and global indices collide (observed natively after a hard kill: resumed rows labelled from '
                           'the totals with global indices past every span)')
            evidence.append(_evidence(info_art, '/total_episodes', total_episodes))
        targets = [('data', 'data/chunk-{:03d}/file-{:03d}.parquet'.format(
            *_next(last['data/chunk_index'], last['data/file_index'], chunks_size)))]
        if _integer(last.get('meta/episodes/chunk_index')) and _integer(last.get('meta/episodes/file_index')):
            targets.append(('metadata', 'meta/episodes/chunk-{:03d}/file-{:03d}.parquet'.format(
                *_next(last['meta/episodes/chunk_index'], last['meta/episodes/file_index'], chunks_size))))
        for key in video_keys:
            chunk, file = last.get(f'videos/{key}/chunk_index'), last.get(f'videos/{key}/file_index')
            if _integer(chunk) and _integer(file):
                targets.append((f'video {key}', _video_rel(template, key, *_next(chunk, file, chunks_size))))
    for role, relative in targets:
        path = work.path(relative)
        if path.is_file():
            reasons.append(f'resume writes its first {role} file to {relative}, which already exists and would be replaced'
                           + ('' if total_episodes == 0 else ' (observed natively after a hard kill: resume replaced the '
                              'data and video files of two committed episodes whose metadata had not been flushed)'))
            evidence.append({'path': str(path), 'sha256': work.artifact(path, 'resume_target')['sha256'],
                             'pointer': 'file resume writes first', 'value': 'exists'})
    if not reasons:
        return None
    return {'code': 'salvage.resume_hazard', 'status': 'error',
            'detected_by': ['episode.declared_totals', 'episode.rows_without_metadata'],
            'message': 'Do not run lerobot-record --resume on this directory as it is: ' + '; '.join(reasons) + '. '
                       f'{LEROBOT} resume takes episode_index and the global index from the info.json totals '
                       f'({LOC["resume_labels"]}) and spans and next files from the last loaded metadata row '
                       f'({LOC["resume_spans"]}); nothing in it reports the result.',
            'remedy': 'Salvage into a separate copy first (see plan.validity.resume_safe), or record further episodes into '
                      'a new dataset.',
            'evidence': evidence[:MAX_EVIDENCE]}
