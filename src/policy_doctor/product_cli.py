"""Small public command surface over the existing evidence engines.

Presentation never promotes an incomplete assessment to PASS. Machine reports and
legacy commands remain available for reproducibility.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import stat
import sys
from pathlib import Path

from . import __version__
from .artifacts import MAX_JSON_BYTES, InputError
from .conversion_fidelity import _read_small, _reference_json

EXIT = {"pass": 0, "healthy": 0, "equal": 0, "warning": 1,
        "error": 1, "changed": 1, "different": 1, "unknown": 3}
KINDS = ("auto", "dataset", "checkpoint", "config", "recording", "training", "upgrade", "sim", "recovery", "groot", "units", "conventions")


def _safe_path(value):
    if not str(value) or not all(c.isprintable() for c in str(value)):
        raise InputError("Paths must be nonempty and contain no terminal control characters.")
    try:
        path = Path(value).expanduser().absolute()
    except RuntimeError as error:
        raise InputError('The selected home directory cannot be resolved; use a direct local path.') from error
    if ".." in path.parts:
        raise InputError("Use a direct local path without parent traversal.")
    for item in (*reversed(path.parents), path):
        try:
            mode = item.lstat().st_mode
        except FileNotFoundError as error:
            raise InputError(f"Not found: {value}. Check the path and rerun.") from error
        if stat.S_ISLNK(mode):
            raise InputError(f"Symlink inputs are refused: {item}. Use the original local path.")
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise InputError(f"Input must be a regular file or directory: {item}.")
    return path


def _document(path):
    raw, _ = _read_small(_safe_path(path), limit=MAX_JSON_BYTES)
    value = _reference_json(raw)
    if type(value) is not dict:
        raise InputError("Expected a JSON object, not an array or scalar.")
    return value


def _output_destination(value):
    if not value or not all(c.isprintable() for c in str(value)):
        raise InputError("--output must name a new file without terminal control characters.")
    try:
        path = Path(value).expanduser().absolute()
    except RuntimeError as error:
        raise InputError('The output home directory cannot be resolved; use a direct local path.') from error
    if ".." in path.parts:
        raise InputError("--output requires a direct path without parent traversal.")
    if not _safe_path(path.parent).is_dir():
        raise InputError("--output parent must be an existing directory.")
    if path.exists() or path.is_symlink():
        raise InputError("--output already exists; choose a new file.")
    return path


def _refuse_report_inside_inputs(args, output):
    """A saved diagnostic must not change the dataset it describes."""
    for value in (args.source, args.reference, args.robot_config, args.converted,
                  args.openpi_source_root, args.runtime_source_root, getattr(args, 'language_catalog', None)):
        if not value or str(value).startswith('hf://'):
            continue
        source = _safe_path(value)
        if source.name in ('info.json', 'modality.json') and source.parent.name == 'meta':
            source = source.parent.parent
        if source.is_dir() and output.is_relative_to(source):
            raise InputError('--output must be outside the input directory: ' + str(source))


def _result_status(report, kind):
    selection = report.get('selected_assessment', {})
    if kind == 'groot' and selection.get('name') == 'groot_consumer_selection':
        return selection['status']
    if kind == 'conventions' and report.get('status') != 'error':
        parts = _assessment_parts(report, kind)
        states = {part.get('status', 'unknown') for part in parts}
        return next((value for value in ('error', 'unknown', 'warning', 'pass') if value in states), 'unknown')
    if "status" in report:
        return report["status"]
    if kind == "config" and type(report.get("equal")) is bool:
        return "equal" if report["equal"] else "different"
    return "unknown"


def _with_options(arguments, *options):
    result = list(arguments)
    position = result.index("--") if "--" in result else len(result)
    flags = {'--video-headers', '--explain', '--json', '--fetch'}
    existing = set(result[:position])
    additions = []
    for option in options:
        if option in flags and option in existing:
            continue
        additions.append(option)
        if option in flags:
            existing.add(option)
    result[position:position] = additions
    from .console_text import display_command

    return display_command(result)


def _detect(source, requested):
    if source.startswith("hf://"):
        if requested not in ("auto", "dataset", "checkpoint"):
            raise InputError("Hub inputs support --kind dataset or --kind checkpoint.")
        return "dataset" if requested == "auto" else requested
    path = _safe_path(source)
    if requested != "auto":
        return requested
    if path.suffix.lower() in (".h5", ".hdf5", ".hdf"):
        return "sim"
    if path.is_dir():
        dataset = (path / "meta/info.json").exists()
        checkpoint = (path / "config.json").exists()
        if dataset and checkpoint:
            raise InputError("Both dataset and checkpoint metadata exist; select --kind dataset or checkpoint.")
        if dataset:
            _document(path / "meta/info.json")
            return "dataset"
        if checkpoint:
            doc = _document(path / "config.json")
            if _checkpoint_config(doc):
                return "checkpoint"
        raise InputError("No supported dataset/checkpoint metadata found. Pass a specific JSON artifact or --kind; see roboclue check --help.")
    if path.suffix == ".jsonl":
        from .fs_guard import open_regular

        with open_regular(path, "recording") as stream:
            header = stream.readline(16385)
        if len(header) > 16384:
            raise InputError("Recording header exceeds the 16 KiB detection bound.")
        doc = _reference_json(header)
        if type(doc) is dict and doc.get("kind") == "flight_recording":
            return "recording"
        raise InputError("Unrecognized JSONL recording. Use a RoboClue flight recording or a specialized advanced command.")
    if path.suffix != ".json":
        raise InputError("Unsupported artifact for automatic checks. Pass dataset/checkpoint metadata, a saved JSON capture/config, or a flight recording; see roboclue check --help.")
    doc = _document(path)
    kind = doc.get("kind")
    if kind == "provenance_register_input":
        return "config"
    if kind == "native_training_target_capture":
        return "training"
    if kind == "native_upgrade_impact":
        return "upgrade"
    if _checkpoint_config(doc):
        return "checkpoint"
    if path.name == "info.json" and path.parent.name == "meta":
        return "dataset"
    if path.name == "modality.json" and path.parent.name == "meta":
        return "groot"
    return "config"


def _checkpoint_config(doc):
    return (type(doc.get("type")) is str
            and type(doc.get("input_features")) is dict
            and type(doc.get("output_features")) is dict)


def _training_pin(args):
    if args.pin_file:
        raw, _ = _read_small(_safe_path(args.pin_file), limit=256)
        try:
            pin = raw.decode("ascii").strip()
        except UnicodeError as error:
            raise InputError("--pin-file must contain one SHA256 pin.") from error
    else:
        pin = args.criterion_sha256
    if pin is not None and not re.fullmatch(r"[0-9a-f]{64}", pin):
        raise InputError("The criterion pin must be exactly 64 lowercase hexadecimal characters.")
    return pin


def _assess(args):
    from . import big_bets_frontdoor as engines

    requested = args.kind
    # These observations explicitly describe a policy checkpoint. Use that
    # role for an otherwise ambiguous Hub URI, without probing either repository
    # or overriding an explicit dataset selection. Fetch still needs opt-in.
    if requested == 'auto' and args.source.startswith('hf://') and any(
            getattr(args, name, None) is not None for name in
            ('instruction', 'instruction_file', 'language_catalog', 'language_command')):
        requested = 'checkpoint'
    kind = _detect(args.source, requested)
    if args.runtime_mode != 'saved' and (kind != 'checkpoint' or args.source.startswith('hf://')):
        raise InputError('--runtime-mode requires a local checkpoint source.')
    if args.runtime_source_root is not None:
        if kind not in ('checkpoint', 'config') or args.source.startswith('hf://'):
            raise InputError('--runtime-source-root requires a local checkpoint, provenance input or exported pipeline contract.')
        _safe_path(args.runtime_source_root)
    if kind == 'checkpoint' and (args.robot_command or args.robot_config or args.openpi_config or args.question != 'auto'):
        raise InputError('Checkpoint image analysis and dataset/robot questions need separate check calls.')
    language_requested = (kind == 'checkpoint' and args.language_command is not None) or any(
        getattr(args, name, None) is not None for name in ('instruction', 'instruction_file', 'language_catalog'))
    if args.language_command is not None and kind not in ('checkpoint', 'config'):
        raise InputError('--launch-command requires a checkpoint or provenance-register input.')
    if language_requested:
        if kind != 'checkpoint':
            raise InputError('Instruction/catalog selectors require a checkpoint source; the dataset check remains separate.')
        if args.max_bytes != 64 * 1024**2 or args.max_rows != 200000 or args.max_files != 1024:
            raise InputError('Checkpoint language intake uses its documented fixed metadata bounds; dataset row/file/byte selectors are inapplicable.')
        for name in ('instruction_file', 'language_catalog'):
            if getattr(args, name, None) is not None:
                _safe_path(getattr(args, name))
    runtime_root = getattr(args, 'runtime_source_root', None)
    if runtime_root is not None:
        if not _safe_path(runtime_root).is_dir():
            raise InputError('--runtime-source-root must be a source or installed-package directory.')
    if kind == 'config' and not args.source.startswith('hf://'):
        from .pipeline_contracts import KINDS as PIPELINE_KINDS
        from .pipeline_contracts import check_pipeline_contract

        document = _document(args.source)
        if document.get('kind') in PIPELINE_KINDS:
            if any(value is not None for value in (args.reference, args.pin_file, args.criterion_sha256,
                    args.robot_command, args.robot_config, args.lerobot_version, args.groot_config,
                    args.groot_config_name, args.groot_version, args.openpi_config, args.openpi_source_root,
                    args.converted, args.require, args.write_copy, args.video_headers, args.language_command)) or args.units or args.fetch or args.question != 'auto' or args.include_candidates:
                raise InputError('Pipeline contracts and dataset/consumer/reference questions need separate check calls.')
            if args.revision != 'main' or args.max_rows != 200000 or args.max_files != 1024 or args.max_bytes != 64 * 1024**2:
                raise InputError('Dataset revisions and row/file/download budgets do not apply to pipeline contracts.')
            return check_pipeline_contract(args.source, runtime_source_root=runtime_root), 'config'
        if runtime_root is not None and document.get('kind') != 'provenance_register_input':
            raise InputError('--runtime-source-root requires an exported openpi_initialization or lerobot_camera_inputs contract.')
    if args.source.startswith("hf://") and not args.fetch:
        raise InputError("Public Hub checks need explicit download permission. Run " +
                         shlex.join(["roboclue", "check", args.source, "--fetch", "--revision", args.revision]) +
                         (" --kind checkpoint" if kind == "checkpoint" else "") +
                         ". Only bounded public artifacts are read; no full videos or model weights.")
    pin = _training_pin(args)
    if kind != "training" and (pin is not None or args.pin_file):
        raise InputError("Criterion pin options apply to saved training captures only.")
    if args.reference and kind not in ("training", "config"):
        raise InputError("--reference applies to config comparisons and saved training captures only.")
    if args.reference:
        _safe_path(args.reference)
    if args.fetch and not args.source.startswith("hf://"):
        raise InputError("--fetch is only for explicit public hf:// inputs; local checks do not download anything.")
    if kind not in ("dataset", "groot", "recovery") and (args.video_headers is not None or args.groot_version):
        raise InputError("--video-headers and --groot-version apply to datasets only.")
    from .product_questions import route

    routed = route(args, kind)
    if routed is not None:
        return routed
    if kind == "dataset":
        source = args.source
        if not source.startswith("hf://"):
            path = _safe_path(source)
            if path.name == "info.json" and path.parent.name == "meta":
                source = str(path.parent.parent)
        config = {"video_headers": bool(args.video_headers), "max_bytes": args.max_bytes,
                  "max_rows": args.max_rows, "max_files": args.max_files}
        if getattr(args, '_dataset_units', False):
            config['units'] = True
        if args.groot_version:
            config["groot_version"] = args.groot_version
        report = engines.health_cards.health_card(source, fetch=args.fetch,
                                                   revision=args.revision, config=config)
    elif kind == "training":
        if not args.reference or pin is None:
            report = {"kind": "training_setup_needed", "status": "unknown",
                      "code": "missing_retained_reference",
                      "message": "This capture needs a reviewed reference and its independently retained pin; historical intent cannot be inferred from current targets.",
                      "source": args.source}
        else:
            _safe_path(args.reference)
            report = engines._check_training_capture(args.reference, pin, args.source)
    elif kind == "recording":
        report = engines.flight_recorder.triage_flight_recording(args.source)
    elif kind == "upgrade":
        report = engines.upgrade_impact.assess_upgrade_impact(args.source)
    elif kind == "checkpoint":
        from .fs_guard import refuse_special_files
        from .hub import check_checkpoint

        if not args.source.startswith("hf://"):
            path = _safe_path(args.source)
            refuse_special_files(path if path.is_dir() else path.parent, "checkpoint")
        report = check_checkpoint(args.source, fetch=args.fetch, revision=args.revision)
        if not args.source.startswith('hf://'):
            from .fastwam_contract import augment_checkpoint_report, check_image_contract

            image_report = check_image_contract(args.source, args.runtime_source_root, args.runtime_mode)
            if image_report is None and (args.runtime_source_root is not None or args.runtime_mode != 'saved'):
                raise InputError('--runtime-mode applies to local FastWAM checkpoints; --runtime-source-root needs one or an exported pipeline contract/provenance input.')
            report = augment_checkpoint_report(report, image_report)
        if language_requested:
            from .language_vocab import augment_report, check_language_vocabulary

            language = check_language_vocabulary(args.source, instruction=args.instruction,
                launch_command=args.language_command, instruction_file=args.instruction_file,
                catalog_directory=args.language_catalog, fetch=args.fetch, revision=args.revision)
            report = augment_report(report, language)
    else:
        document = _document(args.source)
        if args.language_command is not None and document.get('kind') != 'provenance_register_input':
            raise InputError('--launch-command requires a checkpoint or provenance-register input.')
        if document.get('kind') == 'provenance_register_input':
            if args.reference:
                raise InputError('Register inputs are assessed against the curated catalogue; --reference is not supported.')
            from .register import assess_register

            report = assess_register(args.source, launch_command=args.language_command,
                                     runtime_source_root=runtime_root)
        elif args.reference:
            from .compare import compare_configs

            _document(args.reference)
            report = compare_configs(args.reference, args.source)
        else:
            report = {"kind": "config_inspection", "status": "unknown", "code": "missing_config_reference",
                      "source": args.source, "fields_read": len(document),
                      "message": "JSON parsed successfully. Semantic correctness needs a reviewed reference; no training or robot compatibility was assessed."}
            if args.kind == 'auto':
                report.update(code='unrecognized_json_role', message='A JSON object was parsed, but its semantic role was not established. Select an explicit supported artifact kind; no correctness check was performed.')
    return report, kind


def _clean(value, bound=260):
    # Artifacts are untrusted: do not emit terminal controls, newlines, or unbounded text.
    value = " ".join(str(value).split()).replace('policy-deployment-doctor[', 'roboclue[')
    value = "".join(c if c.isprintable() else "?" for c in value)
    if len(value) <= bound:
        return value
    preview = value[:max(0, bound - 20)]
    sentences = list(re.finditer(r'(?<!\d)[.!?](?=\s)', preview))
    if sentences:
        return preview[:sentences[-1].end()] + ' More in --explain.'
    words = preview.rsplit(' ', 1)[0] if ' ' in preview else preview
    return words.rstrip('.,;:') + ' … [see --explain]'


def _readable_message(value):
    """Describe structured evidence without printing a Python dictionary."""
    if isinstance(value, str) and value.lstrip().startswith('{') and len(value) <= 8192:
        try:
            decoded = ast.literal_eval(value)
        except (ValueError, SyntaxError, RecursionError, MemoryError):
            decoded = None
        if isinstance(decoded, dict):
            value = decoded
    if isinstance(value, dict):
        if 'previous' in value and 'current' in value:
            return (f"Stored indices or times jump from {_clean(value['previous'], 40)} to {_clean(value['current'], 40)}"
                    + (f" in episode {_clean(value['episode'], 30)}" if 'episode' in value else '') + '; inspect the cited row evidence.')
        if 'observed' in value and 'declared' in value:
            return f"Observed rows ({_clean(value['observed'], 40)}) disagree with declared metadata ({_clean(value['declared'], 40)})."
        return 'The finding contains structured evidence; inspect the cited fields with --explain.'
    if isinstance(value, str) and 'env_args parsed:' in value:
        return 'Saved environment metadata was parsed. This does not verify the recorded control convention, simulator or training; inspect the declared fields with --explain.'
    return value if isinstance(value, str) else None


def _findings(report):
    data = report.get("report", report)
    findings = list(data.get("findings") or [])
    for component in (data.get("components") or {}).values():
        findings.extend(component.get("findings") or [])
    rank = {"error": 0, "warning": 1, "unknown": 2}
    return sorted((f for f in findings if isinstance(f, dict)),
                  key=lambda f: rank.get(f.get("status"), 3))


def _assessment_parts(report, kind):
    if kind == 'groot' and report.get('selected_assessment', {}).get('name') == 'groot_consumer_selection':
        return [report['selected_assessment']]
    if kind == 'conventions':
        parts = [report.get('components', {}).get('action_conventions', {})]
        context = report.get('components', {}).get('openpi_context')
        if context is not None:
            parts.append(context)
        return parts
    if report.get('kind') == 'declared_robot_interface':
        parts = [report['components']['units']]
        if report['components']['camera_keys'].get('requested'):
            parts.append(report['components']['camera_keys'])
        return parts
    return list(report.get('report', report).get('components', {}).values())


def render(report, *, source=None, kind=None, command=None):
    """Three scoped sentences; retain raw reports and name the selected assessment."""
    if report.get('kind') == 'pipeline_contract_check':
        from .pipeline_contracts import render_pipeline_contract

        return render_pipeline_contract(report)
    if report.get('kind') == 'provenance_register':
        from .register.engine import register_text

        return register_text(report) + '\n'
    status = _result_status(report, kind)
    findings = _findings(report)
    source = source or report.get("request", {}).get("source") or report.get("source", "ARTIFACT")
    kind = kind or {"dataset_health_card": "dataset", "dataset_preflight": "dataset",
                    "native_training_targets": "training", "flight_recording_triage": "recording"}.get(report.get("kind"))
    base = ["roboclue", "check", str(source)]
    repeat_args = []
    original = iter(command or base)
    for value in original:
        if value == "--":
            repeat_args.extend([value, *original])
            break
        if value == "--output":
            next(original, None)
        elif value != "--json" and not value.startswith("--output="):
            repeat_args.append(value)
    repeat = shlex.join(repeat_args)
    option_args = repeat_args[:repeat_args.index("--")] if "--" in repeat_args else repeat_args
    next_command = _with_options(repeat_args, "--explain") if "--explain" not in option_args else repeat
    verdict = {"pass": "PASS — covered checks passed", "healthy": "PASS — covered checks passed",
               "equal": "MATCH — compared configuration values agree", "warning": "REVIEW — warnings in covered checks",
               "different": "CHANGED — compared configuration values differ", "changed": "CHANGED — retained criterion violated",
               "error": "ERROR — a supported contradiction was found", "unknown": "INCOMPLETE — more evidence is needed"}.get(status, _clean(status))
    issue = findings[0] if findings else {}
    problem = _readable_message(issue.get("message") or issue.get("reason") or report.get("message"))
    if problem is not None and not isinstance(problem, str):
        if isinstance(problem, dict) and 'observed' in problem and 'declared' in problem:
            problem = f"Observed rows ({_clean(problem['observed'], 40)}) disagree with declared metadata ({_clean(problem['declared'], 40)})."
        elif isinstance(problem, dict) and 'previous' in problem and 'current' in problem:
            problem = f"Stored indices or times jump from {_clean(problem['previous'], 40)} to {_clean(problem['current'], 40)}; inspect the cited episode evidence."
        else:
            problem = "A supported finding contains structured evidence; inspect its cited fields with --explain."
    if kind == "dataset":
        data = report.get("report", report)
        components = data.get("components") or {}
        passed = [_clean('saved statistics consistency' if name == 'dataset_statistics' else name.replace("_", " "), 45) for name, value in components.items()
                  if value.get("status") in ("pass", "healthy")]
        missing = [_clean(name.replace("_", " "), 45) for name, value in components.items()
                   if value.get("status") == "unknown"]
        if status == "unknown":
            problem = ("Still incomplete: " + ", ".join(missing[:3]) + ". " if missing else
                       "Some evidence is missing; this does not establish a broken dataset. ") + (
                "Covered checks passed: " + ", ".join(passed[:3]) + "." if passed else "")
        video_issues = [f for f in findings if f.get("code") == "video.interval_past_end"]
        if video_issues and status == "error":
            problem = "Episode video timestamps extend past the cited clip; training can read the wrong or missing frame."
        stats_unknown = any("stat" in name for name in missing)
        video_unknown = any("video" in name for name in missing)
        if stats_unknown:
            problem = (problem or "") + " Dataset statistics are missing or unassessed."
        if video_unknown and not any(f.get("status") == "error" for f in findings):
            problem = (problem or "") + " Video evidence is incomplete; decoded images were not checked."
            video_missing = any(f.get('code') == 'video.file_missing' for f in findings)
            if "--video-headers" not in option_args and not video_missing:
                options = ["--video-headers"]
                if source.startswith("hf://") and "--fetch" not in option_args:
                    options.append("--fetch")
                next_command = _with_options(repeat_args, *options)
        if not problem:
            problem = "No contradiction found in the selected metadata/row checks; native training and decoded images remain unassessed."
        if status in ('pass', 'healthy'):
            verdict = 'PASS — selected metadata/row checks only'
            if components.get('dataset_statistics', {}).get('status') == 'pass':
                problem = 'Saved statistics are self-consistent; their values were not compared with dataset rows. The checked row/index invariants passed; robot identity, training compatibility and physical behavior were not verified.'
        comparison = components.get('numeric_rows', {}).get('statistics_comparison', {})
        if components.get('dataset_statistics', {}).get('status') == 'pass':
            from .console_text import statistics_scope

            stats_scope = statistics_scope(comparison, saved_self_consistent=True)
            if status != 'error':
                problem = stats_scope + ' ' + (problem if status == 'unknown' else 'Checked row invariants passed; robot identity and training were not verified.')
        units = components.get('units')
        if isinstance(units, dict) and not any(f.get('status') == 'error' for f in findings):
            from .product_questions import units_view

            view = units_view(units)
            from .console_text import unit_text

            modes = sorted({u for values in view['detected_units'].values() for u in values})
            problem = ('Recorded units: ' + ', '.join(unit_text(value) for value in modes) + '. ' if modes else '') + (
                'Supply the actual robot command and LeRobot version to compare settings.')
            if components.get('dataset_statistics', {}).get('status') == 'pass':
                problem += ' ' + stats_scope
            next_command = _with_options(repeat_args, '--robot-command', 'YOUR_DEPLOY_COMMAND')
        warnings = [f for f in findings if f.get('status') == 'warning']
        if warnings and status not in ('error', 'changed', 'different'):
            headline = _readable_message(warnings[0].get('message') or warnings[0].get('reason')) or 'Review the cited warning.'
            problem = str(warnings[0].get('code', 'warning')) + ': ' + headline + ' ' + problem
            if 'WARNING' not in verdict.upper():
                verdict += '; WARNING in covered evidence'
        elif any(part.get('status') == 'warning' for part in components.values()) and status not in ('error', 'changed', 'different'):
            problem = 'A checked component reports a WARNING; its finding examples require --explain. ' + problem
            if 'WARNING' not in verdict.upper():
                verdict += '; WARNING in covered evidence'
    elif report.get("code") == "missing_retained_reference":
        next_command = "roboclue guard init --help"
    elif kind == "training":
        if status == "pass":
            problem = "Covered targets and masks match the retained reference; other inputs and robot behavior remain unassessed."
        elif status == "error":
            problem = "Captured targets or masks violate the retained intended recipe; inspect the cited values before changing the producer."
    elif report.get("kind") == "config_inspection":
        next_command = (_with_options(repeat_args, '--help') if report.get('code') == 'unrecognized_json_role'
                        else _with_options(repeat_args, "--reference", "REVIEWED_CONFIG.json"))
    elif report.get('kind') == 'provenance_register':
        verdict = ('NO CONTRADICTION — loaded register entries only' if status == 'pass' else verdict)
        problem = problem or report['scope']
        witnesses = issue.get('evidence', [])
        actual = next((w for w in reversed(witnesses) if 'observed' in w), None)
        if actual:
            location = (Path(actual['file']).name + '#' + actual.get('selector', '') if actual.get('file')
                        else actual.get('location', 'selected input'))
            problem += ' Evidence: ' + location + '=' + json.dumps(actual['observed'], ensure_ascii=True)[:80] + '.'
        citation = next(iter(issue.get('source_evidence', [])), {}).get('source')
        if citation:
            problem += ' Source: ' + citation['path'] + ':' + str(citation['line']) + '.'
        if issue.get('first_fix'):
            next_command = _clean(issue['first_fix'], 220)
    elif kind == "config" and type(report.get("equal")) is bool:
        changes = report.get("changes", [])
        if changes:
            first = changes[0]
            problem = (f"Configuration changed at {first['pointer'] or '/'}: "
                       f"{_clean(first.get('before', '(absent)'), 90)} -> {_clean(first.get('after', '(absent)'), 90)}. "
                       "This is a value comparison; causation and training compatibility remain unassessed.")
        else:
            problem = "Compared configuration values match; causation and training compatibility remain unassessed."
    if not problem:
        problem = "Inspect the supported findings and missing evidence; no full training or physical guarantee was assessed."
    from .product_questions import answer

    scoped = answer(report, kind, repeat_args, _clean)
    if scoped is not None:
        verdict, problem, next_command = scoped
    parts = [part for part in _assessment_parts(report, kind) if isinstance(part, dict)]
    scoped_findings = [finding for part in parts for finding in part.get('findings', [])] if parts else findings
    for severity in ('error', 'warning'):
        notable = next((finding for finding in scoped_findings if finding.get('status') == severity), None)
        if notable or any(part.get('status') == severity for part in parts):
            if severity.upper() not in verdict.upper():
                verdict += '; ' + severity.upper() + ' in checked evidence'
                detail = _readable_message(notable.get('message') or notable.get('reason')) if notable else None
                note = ((str(notable.get('code', severity)) + ': ' + detail) if notable and detail else
                        'A checked component reports ' + severity.upper() + '; its finding examples require --explain.')
                problem = note + ' ' + str(problem)
            break
    if not isinstance(problem, str):
        problem = "The finding contains structured evidence; inspect the cited fields with --explain."
    if report.get('image_contract'):
        from .fastwam_contract import evidence_line
        from .language_vocab import language_text

        image = report['image_contract']
        notable = next((f for f in report['findings'] if f['status'] == 'error'), image['findings'][0])
        verdict = report['status'].upper() + ' — checkpoint; image map ' + image['status'].upper()
        problem, next_command = notable['message'], notable.get('remedy', '')
        return (f"Verdict: {verdict}.\nDetails: {_clean(problem, 420)}\n"
                f"Evidence: {_clean(evidence_line(image), 2200)}\nNext: {_clean(next_command, 500)}\n"
                "Scope: static saved files/source; active runtime, native execution and task success unassessed.\n" +
                (language_text(report['language_vocabulary']) + '\n' if report.get('language_vocabulary') else ''))
    language = report.get('language_vocabulary')
    if isinstance(language, dict) and status != 'error':
        language_finding = next((f for f in language.get('findings', []) if f.get('status') == 'warning'), None)
        if language_finding:
            verdict += '; WARNING: instruction outside linked catalog snapshot'
            problem, next_command = language_finding['message'], language_finding['remedy']
        elif language.get('verdict') == 'IN_CATALOG':
            folded = any(row.get('match') == 'lexical' for row in language.get('evaluations', []))
            if folded:
                verdict = 'Instruction has a lexical-only catalog match; native equivalence UNKNOWN'
            else:
                verdict = 'Instruction matches linked catalog'
                if language.get('catalog', {}).get('saved_revision') is None:
                    verdict += '; training revision is unpinned'
            if any(f.get('status') == 'warning' for f in findings):
                verdict += '; independent WARNING remains'
            problem = str(language.get('catalog', {}).get('note', 'Historical coverage is unverified.'))
            problem += ' Native token equivalence and historical training coverage remain UNKNOWN; physical capability is unassessed.'
        else:
            problem = ('Catalog membership: ' + str(language.get('verdict', 'UNKNOWN'))
                       + '. Native token equivalence and historical coverage remain UNKNOWN. '
                       + str(language.get('catalog', {}).get('note', problem)))
    label = 'Problem' if status in ('error', 'different', 'changed') else 'Details'
    return f"Verdict: {verdict}.\n{label}: {_clean(_readable_message(problem), 280)}\nNext: {next_command}\n"


def _parser():
    parser = argparse.ArgumentParser(prog="roboclue", description="Catch robot-learning data and pipeline errors. Start with check or demo.")
    parser.add_argument("--version", action="version", version=f"RoboClue {__version__}")
    commands = parser.add_subparsers(dest="command")
    check = commands.add_parser("check", help="Detect and check a dataset, checkpoint, configuration or saved capture")
    check.add_argument("source", help="Local artifact or hf://ORG/REPO; explicit instruction/catalog selectors select a Hub checkpoint, otherwise use --kind checkpoint for a model")
    check.add_argument("--kind", choices=KINDS, default="auto", help="Resolve ambiguous inputs explicitly")
    check.add_argument("--question", choices=("auto", "dataset", "units", "groot", "recovery", "conversion", "conventions"), default="auto", help="Ask a specific question instead of an unrelated general health check")
    check.add_argument("--robot-command", help="Actual robot command/settings to compare as data; never executed")
    check.add_argument('--instruction', '--task', dest='instruction', help='Declared instruction for a scoped checkpoint task-catalog comparison; no capability claim')
    check.add_argument('--launch-command', dest='language_command', help='Actual literal launch command for instruction/provenance observations; conflicts remain UNKNOWN; never executed')
    check.add_argument('--instruction-file', help='Static JSON instruction list or literal Python TASK_INSTRUCTIONS list; never executed')
    check.add_argument('--language-catalog', help='Explicit local dataset directory containing meta/tasks.parquet or tasks.jsonl; historical linkage remains unverified')
    check.add_argument("--robot-config", help="Actual robot JSON/YAML configuration")
    check.add_argument("--lerobot-version", help="Caller-declared fleet LeRobot version for a robot command/config; not this tool's Python environment or an attestation of installed source")
    check.add_argument("--units", action="store_true", help="Identify recorded units; robot compatibility requires its command/config")
    check.add_argument("--groot-config", help="Actual training config .py text; parsed, never executed")
    check.add_argument("--groot-config-name", help="Selected GR00T config class or builtin name")
    check.add_argument("--openpi-config", help="Selected builtin OpenPI TrainConfig; convention check only")
    check.add_argument("--openpi-source-root", help="Pinned OpenPI source checkout for the declared config")
    check.add_argument('--runtime-source-root', help='Actual consumer source directory for supported checkpoint, provenance or initialization/camera contracts; read without execution')
    check.add_argument('--runtime-mode', choices=('saved', 'training'), default='saved', help='Saved inference/eval processor or effective fresh-training config; resume overrides unassessed')
    check.add_argument("--converted", help="Candidate HDF5/export to compare with the source recording")
    check.add_argument("--write-copy", metavar="NEWDIR", help="Explicitly apply a supported recording-recovery plan to a new directory, preserve the original and recheck the copy")
    check.add_argument("--include-candidates", action="store_true", help="With --write-copy, reconstruct supported complete episodes whose metadata was not saved; requires the recovery extra")
    check.add_argument("--require", choices=("deployment-units", "groot", "conventions", "recovery", "sim"), help="Require the named assessment; missing or unknown required evidence never clears a gate")
    check.add_argument("--fail-on", choices=("error", "warning", "unknown"), help="Finding threshold; --require additionally enforces required coverage")
    check.add_argument("--reference", "--contract", dest="reference", help="Reviewed saved reference; never generated from the candidate")
    pin = check.add_mutually_exclusive_group()
    pin.add_argument("--criterion-sha256", help="Independently retained training reference SHA256")
    pin.add_argument("--pin-file", help="File containing the independently retained training reference SHA256")
    check.add_argument("--fetch", action="store_true", help="Allow bounded public Hub intake; never downloads full videos or weights")
    check.add_argument("--video-headers", action=argparse.BooleanOptionalAction, default=None, help="Inspect bounded video headers (default on for local v3 recovery and routing; generic health coverage is reported separately); decoded frames remain unassessed")
    check.add_argument("--revision", default="main")
    check.add_argument("--groot-version", choices=("n1.5", "n1.7"), help="Explicit GR00T dataset contract")
    check.add_argument("--max-rows", type=int, default=200000)
    check.add_argument("--max-files", type=int, default=1024)
    check.add_argument("--max-bytes", type=int, default=64 * 1024**2)
    check.add_argument("--json", action="store_true", help="Print the original machine report; preserve status and evidence")
    check.add_argument("--explain", action="store_true", help="Show the detailed evidence and scope after the three-line answer")
    check.add_argument("--output", help="Save the original JSON report to a new file")
    commands.add_parser("demo", add_help=False, help="See a small fault corrected and verified on CPU, offline")
    commands.add_parser("guard", add_help=False, help="Guard intended training inputs or check a recording before resume")
    commands.add_parser("mcp", add_help=False, help="Configure or run the read-only MCP server")
    commands.add_parser("advanced", add_help=False, help="Specialist diagnostics and compatibility commands")
    return parser


def _advanced(arguments):
    if arguments[:1] == ['versions']:
        from .framework_versions import collect_framework_versions, render_framework_versions

        parser = argparse.ArgumentParser(prog='roboclue advanced versions', description='Identify installed framework metadata without claiming runtime compatibility.')
        parser.add_argument('--json', action='store_true')
        args = parser.parse_args(arguments[1:])
        result = collect_framework_versions()
        print(json.dumps(result, indent=2) if args.json else render_framework_versions(result))
        return 0
    if arguments[:1] == ['rules']:
        from .rule_catalog import get_rule, render_catalog, render_rule, rule_catalog

        parser = argparse.ArgumentParser(prog='roboclue advanced rules', description='Browse supported diagnostic rules and their evidence boundaries.')
        parser.add_argument('rule_id', nargs='?')
        parser.add_argument('--json', action='store_true')
        args = parser.parse_args(arguments[1:])
        try:
            result = get_rule(args.rule_id) if args.rule_id else rule_catalog()
            print(json.dumps(result, indent=2) if args.json else render_rule(result) if args.rule_id else render_catalog())
            return 0
        except InputError as error:
            print('Input: ' + str(error), file=sys.stderr)
            return 2
    from . import big_bets_frontdoor as frontdoor
    from .cli import main as legacy_main

    specialized = {"start", "demo-native", "preflight", "monitor", "upgrade-watch", "health-card",
                   "flight-triage", "upgrade-impact", "check-training-targets"}
    if not arguments or arguments[0] in ("-h", "--help"):
        print("RoboClue advanced diagnostics\n\nEvidence workflows: " + ", ".join(sorted(specialized | {'rules', 'versions'})) +
              "\nOther specialist commands are listed below. Prefix them with roboclue advanced.\n")
        return legacy_main(["--help"], prog='roboclue advanced', brand='RoboClue', command_prefix=('roboclue', 'advanced'))
    if arguments[0] in specialized:
        if arguments[0] == 'preflight':
            return legacy_main(arguments, prog='roboclue advanced', brand='RoboClue', command_prefix=('roboclue', 'advanced'))
        return frontdoor.main(arguments)
    return legacy_main(arguments, prog='roboclue advanced', brand='RoboClue', command_prefix=('roboclue', 'advanced'))


def _main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["demo"]:
        from .product_demo import main as demo_main

        return demo_main(arguments[1:])
    if arguments[:1] == ["guard"]:
        from .product_guard import main as guard_main

        return guard_main(arguments[1:])
    if arguments[:1] == ["advanced"]:
        return _advanced(arguments[1:])
    if arguments[:1] == ["mcp"]:
        from .product_mcp import main as mcp_main

        return mcp_main(arguments[1:])
    # Accept old invocations for existing scripts; the public help advertises five verbs.
    legacy = {"start", "demo-native", "preflight", "monitor", "upgrade-watch", "health-card",
              "flight-triage", "upgrade-impact", "check-training-targets"}
    if arguments and arguments[0] in legacy:
        return _advanced(arguments)
    parser = _parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        output = None
        if args.output:
            output = _output_destination(args.output)
            _refuse_report_inside_inputs(args, output)
        report, kind = _assess(args)
        if output:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        framework_context = None
        framework = {'units': 'lerobot', 'recovery': 'lerobot', 'groot': 'groot', 'sim': 'isaaclab', 'conventions': 'openpi'}.get(kind)
        if framework and not args.json:
            from .framework_versions import collect_framework_versions, compact_framework_context

            framework_context = compact_framework_context(collect_framework_versions(), framework)
        if args.json:
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            rendered = render(report, source=args.source, kind=kind, command=["roboclue", *arguments])
            print(rendered, end="")
            if args.explain:
                from .big_bets_frontdoor import LIMITS

                print("\nEvidence and scope:\n" + json.dumps(report, indent=2, allow_nan=False) + "\n" + LIMITS)
                if framework_context:
                    print('\nInstalled framework metadata context (not a runtime validation):\n' + json.dumps(framework_context, indent=2))
        status = _result_status(report, kind)
        required = report.get('coverage', {}).get('required_assessments', {}).get(args.require, {})
        selected_findings = (report.get('components', {}).get('action_conventions', {}).get('findings', [])
                             if kind == 'conventions' else _findings(report))
        if report.get('kind') == 'declared_robot_interface':
            parts = report['components']
            selected_findings = list(parts['units'].get('findings', []))
            if parts['camera_keys'].get('requested'):
                selected_findings.extend(parts['camera_keys'].get('findings', []))
        ambiguous_groot = kind == 'groot' and report.get('selected_assessment', {}).get('name') == 'groot_consumer_selection'
        if ambiguous_groot:
            selected_findings = []  # Per-version results remain evidence, not a selected consumer gate.
        finding_statuses = {f.get('status') for f in selected_findings if isinstance(f, dict)} | {status}
        selected_components = _assessment_parts(report, kind)
        finding_statuses.update(part.get('status') for part in selected_components if isinstance(part, dict))
        if args.fail_on:
            failing = {'error': {'error', 'different', 'changed'},
                       'warning': {'error', 'different', 'changed', 'warning'},
                       'unknown': {'error', 'different', 'changed', 'warning', 'unknown'}}[args.fail_on]
            if finding_statuses & failing:
                return 1
            if required.get('status') == 'unknown' or status == 'unknown' and (
                args.require or args.robot_command or args.robot_config or args.question != 'auto' or kind == 'training' or args.reference or ambiguous_groot
            ):
                return 3
            return 0
        if required.get('status') == 'unknown' and status != 'error':
            return 3
        if finding_statuses & {'error', 'different', 'changed', 'warning'}:
            return 1
        return EXIT.get(status, 3)
    except BrokenPipeError:
        raise
    except (InputError, OSError, ValueError, ImportError) as error:
        if args.json:
            print(json.dumps({'status': 'input_error', "error": {"code": "invalid_input", "message": str(error)}}))
        else:
            message = str(error).replace('policy-deployment-doctor[', 'roboclue[')
            print("Input: " + _clean(message, 600), file=sys.stderr)
        return 2


def main(argv=None):
    if argv is not None:
        return _main(argv)
    try:
        try:
            return _main()
        finally:
            # The stdio MCP transport closes stdout when its client disconnects;
            # a closed stream holds nothing to flush and is a normal shutdown.
            if not sys.stdout.closed:
                sys.stdout.flush()
    except BrokenPipeError:
        # A downstream reader cancelled the stream. Keep cancellation nonzero
        # and prevent Python's shutdown flush from emitting a second exception.
        descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(descriptor, sys.stdout.fileno())
        finally:
            os.close(descriptor)
        return 141


if __name__ == "__main__":
    raise SystemExit(main())
