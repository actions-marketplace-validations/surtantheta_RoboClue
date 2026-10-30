"""Opt-in gate bound to the recording command's actual local dataset.

This gate models the retained LeRobot v3 recording layout. It does not execute
the recorder until the original directory clears the required bounded checks.
The advisory directory lock coordinates RoboClue wrappers, not other writers.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from .artifacts import InputError


def bind_recording_command(dataset, command):
    """Require an unambiguous, matching root and affirmative resume selector."""
    from .product_cli import _safe_path

    root = _safe_path(dataset)
    if not root.is_dir():
        raise InputError('Recording resume needs the actual dataset directory, not a metadata file.')
    if not isinstance(command, (list, tuple)) or not command or not all(
        isinstance(t, str) and t and all(c.isprintable() for c in t) for t in command
    ):
        raise InputError('Supply the recorder argument vector after --; shell strings are not executed.')
    if Path(command[0]).name != 'lerobot-record':
        raise InputError('Resume currently supports the lerobot-record console command only; no arbitrary trainer or shell is wrapped.')
    values = {}
    index = 1
    while index < len(command):
        token = command[index]
        key, sep, value = token.partition('=')
        if key in ('--config', '--config_path', '--dataset'):
            raise InputError('Config-file and aggregate dataset overrides are ambiguous here; supply explicit --dataset.root and --resume=true.')
        if key in ('--dataset.root', '--resume'):
            if key in values:
                raise InputError('Repeated ' + key + ' is refused; the checked recording root must be unambiguous.')
            if not sep:
                index += 1
                if index >= len(command) or command[index].startswith('--'):
                    raise InputError(key + ' needs an explicit value; use --resume=true and --dataset.root=PATH.')
                value = command[index]
            values[key] = value
        elif token == '--' or not token.startswith('--') or not sep:
            raise InputError('Use --key=value for other recorder options; positional values, shell syntax and ambiguous options are not supported by this wrapper.')
        index += 1
    if values.get('--resume', '').lower() != 'true':
        raise InputError('The wrapped command must explicitly declare --resume=true; this gate is for existing recordings only.')
    declared = values.get('--dataset.root')
    if not declared:
        raise InputError('The wrapped command needs --dataset.root=PATH; a repo ID or inferred cache root cannot bind this gate.')
    if declared.startswith('~'):
        raise InputError('Use an absolute or direct relative recorder root without tilde; native Path handling may not expand it like RoboClue.')
    import yaml

    try:
        decoded = yaml.safe_load(declared)
    except (yaml.YAMLError, RecursionError) as exc:
        raise InputError('The recorder root is not an unambiguous native YAML string; use a direct absolute path.') from exc
    if not isinstance(decoded, str) or decoded != declared or decoded.startswith('include'):
        raise InputError('Native argument parsing changes or includes this root value. Use a direct absolute path with no outer YAML quotes, comments or include directive.')
    if _safe_path(declared) != root:
        raise InputError('The wrapped --dataset.root differs from the checked dataset. Check the exact directory the recorder will resume.')
    return root, list(command)


@contextmanager
def recording_lock(root):
    """Hold a cooperative POSIX directory lock without changing dataset bytes."""
    try:
        import fcntl
    except ImportError as exc:
        raise InputError('Recording resume locking is currently supported on POSIX systems only.') from exc
    descriptor = os.open(root, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InputError('Another RoboClue resume wrapper holds this dataset. Wait for that recording to finish.') from exc
        yield
    finally:
        os.close(descriptor)


def assess_resume(dataset, command, *, max_rows=200000, max_files=1024):
    from .framework_versions import collect_framework_versions, compact_framework_context
    from .product_cli import _document
    from .salvage import check_salvage

    root, argv = bind_recording_command(dataset, command)
    if _document(root / 'meta/info.json').get('codebase_version') != 'v3.0':
        raise InputError('Resume gating is qualified for the native LeRobot v3.0 recording layout only; do not relabel unsupported data.')
    report = check_salvage(str(root), max_rows=max_rows, max_files=max_files, video_headers=True)
    current = (report.get('plan') or {}).get('validity', {}).get('current_directory_resume_status', 'unknown')
    complete = report.get('coverage', {}).get('complete') is True
    ready = report.get('status') == 'pass' and complete and current == 'no_hazard_detected'
    code = 0 if ready else 1 if report.get('status') in ('error', 'warning') or current == 'unsafe' else 3
    return {
        'schema_version': 1, 'kind': 'recording_resume_gate',
        'status': 'pass' if ready else 'blocked', 'assessment_status': report.get('status', 'unknown'),
        'dataset': str(root), 'required_check': 'original_recording_integrity',
        'current_directory_resume_status': current, 'coverage_complete': complete,
        'command': argv, 'launched': False, 'exit_code': code, 'report': report,
        'framework_context': compact_framework_context(collect_framework_versions(), 'lerobot'),
        'scope': 'Bounded original LeRobot v3 recording rows, metadata, resume hazards and MP4 headers before the declared command starts. No decoded frames, robot calibration, physical safety, task success or arbitrary trainer compatibility is established. Do not run other writers; the advisory lock only coordinates RoboClue resume wrappers.',
    }


def run_recorder(command):
    """Forward cancellation while retaining the lock until the recorder exits."""
    child = None
    pending = []
    previous = {}

    def forward(signum, _frame):
        pending.append(signum)
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        if pending:
            return 128 + pending[0]
        child = subprocess.Popen(command, start_new_session=True)
        for signum in pending:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
        code = child.wait()
        return code if code >= 0 else 128 - code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _resume_problem(findings):
    """Lead with an observed resume hazard before secondary metadata symptoms."""
    relevant = [f for f in findings if f.get('status') in ('error', 'unknown', 'warning')]
    hazard = next((f for f in relevant if f.get('code') == 'salvage.resume_hazard'), None)
    selected = hazard or next(iter(relevant), None)
    return selected.get('message', 'Required evidence is incomplete.') if selected else 'Required evidence is incomplete.'


def main(argv=None):
    parser = argparse.ArgumentParser(prog='roboclue guard resume', description='Check the exact recording directory before starting lerobot-record --resume.')
    parser.add_argument('dataset', help='Existing local LeRobot v3 recording directory')
    parser.add_argument('--max-rows', type=int, default=200000)
    parser.add_argument('--max-files', type=int, default=1024)
    parser.add_argument('--dry-run', action='store_true', help='Check and bind the command, without launching it')
    parser.add_argument('--json', action='store_true', help='Machine evidence; requires --dry-run')
    arguments = list(sys.argv[1:] if argv is None else argv)
    separator = arguments.index('--') if '--' in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1:]
    try:
        if args.json and not args.dry_run:
            raise InputError('--json requires --dry-run; recorder stdout is not a JSON report.')
        root, command = bind_recording_command(args.dataset, command)
        with recording_lock(root):
            result = assess_resume(root, command, max_rows=args.max_rows, max_files=args.max_files)
            if args.json:
                print(json.dumps(result, allow_nan=False))
            elif result['exit_code']:
                print('Resume blocked: the original directory did not clear every required recording check.', flush=True)
                findings = result['report'].get('findings', [])
                problem = _resume_problem(findings)
                print('Problem: ' + str(problem), flush=True)
                print('Next: roboclue check ' + shlex.quote(str(root)) + ' --question recovery --video-headers --explain', flush=True)
            else:
                print('Recording integrity checks cleared for the declared root; ' + ('dry run, recorder not started.' if args.dry_run else 'starting the requested recorder.'), flush=True)
                print('Scope: rows, metadata, resume hazards and MP4 headers; no robot or physical validation.', flush=True)
            if result['exit_code'] or args.dry_run:
                return result['exit_code']
            return run_recorder(command)
    except (InputError, OSError, ValueError, ImportError) as exc:
        if args.json:
            print(json.dumps({'kind': 'recording_resume_gate', 'status': 'input_error', 'launched': False, 'error': str(exc)}), file=sys.stderr)
        else:
            print('Resume not started: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
