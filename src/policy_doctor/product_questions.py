"""Select existing diagnostic engines by the question and declared consumer.

No recipe, deployment command or config is executed. The returned engine report
retains its own verdict; presentation explains the narrower boundary it answers.
"""
from __future__ import annotations

import shlex

from .artifacts import InputError

QUESTION_KINDS = ('sim', 'recovery', 'groot', 'units', 'conventions')
QUESTIONS = ('auto', 'dataset', 'units', 'groot', 'recovery', 'conversion', 'conventions')


def dataset_root(source):
    from .product_cli import _safe_path

    path = _safe_path(source)
    if path.name in ('info.json', 'modality.json') and path.parent.name == 'meta':
        path = path.parent.parent
    if not path.is_dir():
        raise InputError('This question needs a local dataset directory, or its meta/info.json or meta/modality.json.')
    from .fs_guard import refuse_special_files

    refuse_special_files(path, 'dataset')
    return path


def route(args, kind):
    """Return (original engine report, route), or None for the generic front door."""
    from .product_cli import _document, _safe_path

    explicit = []
    if getattr(args, 'include_candidates', False) and not getattr(args, 'write_copy', None):
        raise InputError('--include-candidates needs --write-copy NEWDIR; candidates are never added to the original dataset.')
    if getattr(args, 'lerobot_version', None) is not None and not (args.robot_command or args.robot_config):
        raise InputError('--lerobot-version needs the actual --robot-command or --robot-config; a version label alone does not describe a consumer.')
    if getattr(args, 'write_copy', None) is not None:
        if args.video_headers is False:
            raise InputError('--write-copy requires local video-header checks; omit --no-video-headers.')
        explicit.append('recovery')
    if args.require:
        if args.require == 'deployment-units':
            if not args.robot_command and not args.robot_config:
                raise InputError('--require deployment-units needs the actual --robot-command or --robot-config; dataset-only unit identification is not deployment comparison.')
            explicit.append('units')
        else:
            explicit.append(args.require)
    if args.robot_command is not None or args.robot_config is not None or args.units:
        explicit.append('units')
    if args.groot_config is not None or args.groot_config_name is not None or args.groot_version is not None:
        explicit.append('groot')
    if args.openpi_config is not None or args.openpi_source_root is not None:
        explicit.append('conventions')
    if args.converted is not None:
        explicit.append('sim')
    requested = {'conversion': 'sim', 'dataset': 'dataset'}.get(args.question, args.question)
    if requested != 'auto':
        explicit.append(requested)
    if kind in QUESTION_KINDS:
        explicit.append(kind)
    if len(set(explicit)) > 1:
        raise InputError('Choose one question/consumer per check; units, GR00T, OpenPI and conversion selectors cannot silently override each other.')
    selected = next(iter(explicit), kind)
    if selected in QUESTION_KINDS and (args.reference or args.pin_file or args.criterion_sha256):
        raise InputError('--reference and criterion pins belong to config/training checks; use --converted for a simulator source/export comparison.')
    if selected in QUESTION_KINDS and args.source.startswith('hf://'):
        raise InputError('This consumer-specific check needs the local dataset or recording. Materialize the source with its native workflow; no consumer intent is inferred from a Hub label.')
    if selected == 'sim':
        from .sim_guards import check_sim_provenance

        if args.converted:
            _safe_path(args.converted)
        return check_sim_provenance(args.source, converted=args.converted, max_rows=args.max_rows), 'sim'
    if selected not in ('dataset', *QUESTION_KINDS):
        return None
    if args.source.startswith('hf://'):
        return None
    root = dataset_root(args.source)
    info = _document(root / 'meta/info.json')
    if getattr(args, 'write_copy', None) is not None:
        if not str(info.get('codebase_version', '')).startswith('v3'):
            raise InputError('--write-copy currently supports native LeRobot v3 recovery only; changing a dataset version label is not a repair.')
        from .recovery_copy import write_recovery_copy

        return write_recovery_copy(str(root), args.write_copy, max_rows=args.max_rows,
                                   max_files=args.max_files, include_candidates=getattr(args, 'include_candidates', False)), 'recovery'
    if selected in ('dataset', 'recovery') and str(info.get('codebase_version', '')).startswith('v3'):
        # Automatic and explicit recovery share the header default. An explicit
        # opt-out remains scoped to identities and cannot establish video health.
        from .salvage import check_salvage

        recovery = check_salvage(str(root), max_rows=args.max_rows, max_files=args.max_files,
                                 video_headers=args.video_headers is not False, compare_stats=selected == 'recovery')
        # Do not discard a specialist error (or recording leftovers) merely
        # because its code is absent from a front-door allowlist.
        damaged = any(f.get('status') == 'error' or f.get('code') == 'salvage.leftovers'
                      for f in recovery.get('findings', []))
        metadata_review = recovery.get('assessment_kind') in ('metadata_rows_mismatch', 'metadata_rows_unresolved')
        pointer_repair = bool((recovery.get('plan') or {}).get('pointer_corrections'))
        if selected == 'recovery' or damaged or metadata_review or pointer_repair:
            if selected != 'recovery' and recovery.get('coverage', {}).get('complete') is True and (
                recovery.get('plan') or {}).get('validity', {}).get('current_directory_resume_status') == 'no_hazard_detected':
                recovery = check_salvage(str(root), max_rows=args.max_rows, max_files=args.max_files,
                                         video_headers=args.video_headers is not False)
            elif selected != 'recovery' and recovery.get('statistics_comparison', {}).get('reason') == 'Explicitly disabled.':
                recovery['statistics_comparison']['reason'] = (
                    'Resolve the incomplete or inconsistent recording ledger before comparing numeric values.')
            return recovery, 'recovery'
    if selected == 'recovery':
        raise InputError('Recording recovery currently supports native LeRobot v3 datasets only; changing the version label is not a repair.')
    if selected == 'dataset' and args.question == 'auto':
        if (root / 'meta/modality.json').is_file():
            selected = 'groot'
        elif str(info.get('robot_type', '')).startswith(('so100', 'so101', 'koch', 'omx', 'lekiwi')):
            # Preserve the general health assessment as well as identifying the
            # recorded units. No deployment consumer was provided to compare.
            args._dataset_units = True
            return None
    if selected == 'units':
        from .robot_contract import check_declared_camera_keys, validate_robot_declaration
        from .units_manifest import units_manifest

        if args.robot_command or args.robot_config:
            validate_robot_declaration(robot_command=args.robot_command, robot_config=args.robot_config)
        if args.robot_command:
            tokens = shlex.split(args.robot_command)
            if any(t.startswith('--robot.use_degree') and not t.startswith('--robot.use_degrees=') and t != '--robot.use_degrees' for t in tokens):
                raise InputError('Unknown robot units option; use --robot.use_degrees=true or --robot.use_degrees=false in the deployment command.')
        # The command is parsed as data by the existing engine, never executed.
        report = units_manifest(str(root), robot_command=args.robot_command, robot_config=args.robot_config,
                                max_rows=args.max_rows, max_files=args.max_files)
        if args.robot_command or args.robot_config:
            from .consumer_version import apply_consumer_version

            report = apply_consumer_version(report, getattr(args, 'lerobot_version', None), require_version=True)
        deployment = report.get('manifest', {}).get('deployment')
        if isinstance(deployment, dict) and deployment.get('unknown') and (args.robot_command or args.robot_config):
            raise InputError('The declared deployment could not be interpreted: ' + str(deployment['unknown']) + '. Check the robot type and flags; no deployment gate passed.')
        if args.robot_command or args.robot_config:
            cameras = check_declared_camera_keys(str(root), robot_command=args.robot_command, robot_config=args.robot_config,
                                                lerobot_version=getattr(args, 'lerobot_version', None), require_version=True)
            states = [report['status']]
            if cameras.get('requested'):
                states.append(cameras.get('status', 'unknown'))
            rank = {'error': 0, 'unknown': 1, 'warning': 2, 'pass': 3}
            state = min(states, key=lambda value: rank.get(value, 2))
            return {'schema_version': 1, 'kind': 'declared_robot_interface', 'status': state,
                    'source': str(root), 'components': {'units': report, 'camera_keys': cameras},
                    'scope': 'Recorded units and explicitly declared camera keys only. Image size/rate, joint order, policy input mapping, native execution and physical behavior remain unassessed.'}, 'units'
        return report, 'units'
    if selected == 'groot':
        from .groot_contract import check_groot_dataset_versions

        if args.groot_config:
            _safe_path(args.groot_config)
        report = check_groot_dataset_versions(str(root), args.groot_config,
                    config_name=args.groot_config_name, groot_version=args.groot_version)
        statuses = {part.get('status', 'unknown') for part in report.get('versions', {}).values()}
        if not args.groot_version and report.get('mode') == 'both' and len(statuses) > 1:
            report['selected_assessment'] = {'name': 'groot_consumer_selection', 'status': 'unknown',
                'reason': 'GR00T profiles disagree and no consumer version was selected.',
                'scope': 'Per-version findings are retained; select the actual consumer with --groot-version.'}
        return report, 'groot'
    if selected == 'conventions':
        from .preflight import preflight

        if args.openpi_source_root:
            _safe_path(args.openpi_source_root)
        report = preflight(str(root), openpi_config=args.openpi_config,
                           openpi_source_root=args.openpi_source_root, groot_contract=False,
                           video_headers=False, max_rows=args.max_rows, max_files=args.max_files)
        part = report.get('components', {}).get('action_conventions')
        if part is None:
            raise InputError('The requested OpenPI convention check is absent; supply both --openpi-config and --openpi-source-root for the supported pinned source.')
        context = report.get('components', {}).get('openpi_context')
        parts = [part] + ([context] if isinstance(context, dict) else [])
        unresolved = [f.get('message') or f.get('reason') or f.get('code')
                      for component in parts for f in component.get('findings', []) if f.get('status') == 'unknown']
        statuses = {component.get('status', 'unknown') for component in parts}
        selected_status = next((value for value in ('error', 'unknown', 'warning', 'pass') if value in statuses), 'unknown')
        report['selected_assessment'] = {'name': 'action_conventions', 'status': selected_status,
                                         'delta_transform_status': part.get('verdicts', {}).get('consumer', 'unknown'),
                                         'raw_input_status': context.get('status', 'unknown') if isinstance(context, dict) else 'not_assessed',
                                         'scope': 'The consumer verdict covers the delta-transform choice only. The separate OpenPI context checks statically declared repack keys and reports dataset format and consumer lockfile facts; installed-loader compatibility, image contents and native training are not assessed.',
                                         'unresolved': unresolved}
        if args.require == 'conventions':
            report.setdefault('coverage', {}).setdefault('required_assessments', {})['conventions'] = {
                'status': 'error' if selected_status == 'error' else 'unknown' if unresolved else selected_status,
                'unresolved': unresolved,
                'scope': 'Declared consumer choice plus requested convention evidence; undecided fingerprints are not established by a matching delta-transform choice.'}
        # Keep the full preflight and all its findings in JSON. The renderer
        # selects this component, not a generic unrelated video result.
        return report, 'conventions'
    return None


def units_view(report):
    """Bounded named units and comparison boundary; never deployment approval."""
    manifest = report.get('manifest') or {}
    detected = {}
    for name, feature in (manifest.get('features') or {}).items():
        units = sorted({str(c.get('unit')) for c in feature.get('coordinates', []) if c.get('unit')})
        detected[str(name)[:96]] = units[:8]
        if len(detected) >= 8:
            break
    deployment = manifest.get('deployment')
    return {'check': 'dataset_units_consistency', 'reported_status': report.get('status', 'unknown'),
            'detected_units': detected,
            'compared_with': deployment.get('robot_type') if isinstance(deployment, dict) else None,
            'deployment_comparison': 'declared_robot_command_or_config' if deployment is not None else 'not_assessed',
            'comparison_statement': 'Compared with the declared robot settings only.' if deployment is not None else 'No robot command/config supplied; deployment was not compared.',
            'scope': 'Inferred recorded numeric units. With a declared consumer, compare its configured units assuming policy outputs retain dataset units; policy processors, calibration and physical deployment remain unassessed.'}


def _ranges(ranges):
    shown = []
    for low, high in (ranges or [])[:8]:
        shown.append(str(low) if low == high else f'{low}-{high}')
    return ', '.join(shown) + (' ... (more ranges in evidence)' if len(ranges or []) > 8 else '')


def _rerun(arguments, *options):
    from .product_cli import _with_options

    return _with_options(arguments, *options)


def _unit_correction_command(report, arguments):
    manifest = report.get('manifest', {})
    deployment = manifest.get('deployment') or {}
    if set(deployment.get('arms', {})) != {''}:
        return None
    modes = {c.get('unit') for f in manifest.get('features', {}).values()
             for c in f.get('coordinates', []) if c.get('motor') not in (None, 'gripper') and c.get('status') == 'decided'}
    if modes not in ({'RANGE_M100_100'}, {'DEGREES'}):
        return None
    value = 'false' if modes == {'RANGE_M100_100'} else 'true'
    args = list(arguments)
    position = next((i for i, token in enumerate(args) if token == '--robot-command' or token.startswith('--robot-command=')), None)
    if position is None:
        return None
    joined = args[position].startswith('--robot-command=')
    original = args[position].split('=', 1)[1] if joined else args[position + 1]
    command = shlex.split(original)
    existing = next((i for i, token in enumerate(command) if token == '--robot.use_degrees' or token.startswith('--robot.use_degrees=')), None)
    if existing is None:
        command.append('--robot.use_degrees=' + value)
    elif command[existing] == '--robot.use_degrees':
        command[existing:existing + 2] = ['--robot.use_degrees=' + value]
    else:
        command[existing] = '--robot.use_degrees=' + value
    updated = shlex.join(command)
    if joined:
        args[position] = '--robot-command=' + updated
    else:
        args[position + 1] = updated
    from .console_text import display_command

    return display_command(args)


def answer(report, kind, arguments, clean):
    """Question-specific three-line content, without rewriting raw statuses."""
    state = report.get('status', 'unknown')
    findings = sorted(report.get('findings', []), key=lambda f: {'error': 0, 'warning': 1, 'unknown': 2}.get(f.get('status'), 3))
    first = findings[0] if findings else {}
    message = first.get('message') or first.get('reason') or 'No supported contradiction found in the selected checks.'
    if not isinstance(message, str):
        message = 'Inspect the cited values in the detailed evidence; the finding has structured details.'
    next_command = _rerun(arguments, '--explain') if '--explain' not in arguments else _rerun([a for a in arguments if a != '--explain'], '--json')
    if kind == 'units':
        from .console_text import unit_text

        units = report.get('components', {}).get('units', report)
        camera = report.get('components', {}).get('camera_keys', {})
        view = units_view(units)
        labels = sorted({unit for values in view['detected_units'].values() for unit in values})
        label = ', '.join(unit_text(value) for value in labels) or 'not determined'
        consumer = view['compared_with']
        if not consumer:
            verdict = f'Recorded units: {label}; deployment NOT ASSESSED'
            problem = 'These units were inferred from recorded values. Supply the actual robot command and its LeRobot version to compare them.'
            # An exact command with a clearly marked input placeholder; no robot
            # configuration is invented from the dataset's robot label.
            next_command = _rerun(arguments, '--robot-command', 'YOUR_DEPLOY_COMMAND')
        else:
            unit_state = units.get('status', 'unknown')
            unit_findings = units.get('findings', [])
            verdict = f'Unit comparison with {consumer}: {unit_state.upper()}'
            unit_issue = next((f for f in unit_findings if f.get('status') in ('error', 'unknown', 'warning')), {})
            problem = str(unit_issue.get('message') or unit_issue.get('reason') or 'The declared robot settings were compared with the recorded units.')
            if not unit_findings and unit_state != 'pass':
                problem = f'Units assessment: {unit_state.upper()}; finding examples are unavailable here. Inspect the detailed evidence.'
            if any(f.get('code') == 'units.joint_mode_mismatch' for f in unit_findings) or unit_state == 'pass':
                body_modes = sorted({c.get('unit') for feature in units.get('manifest', {}).get('features', {}).values()
                                     for c in feature.get('coordinates', []) if c.get('motor') not in (None, 'gripper') and c.get('unit')})
                configured = sorted({a.get('mode') for a in units.get('manifest', {}).get('deployment', {}).get('arms', {}).values() if a.get('mode')})
                problem = (f"Recorded joints: {', '.join(unit_text(value) for value in body_modes) or 'not determined'}; "
                           f"robot expects: {', '.join(unit_text(value) for value in configured) or 'not determined'}. "
                           'This compares settings, assuming the policy preserves recorded units.')
            arms = units.get('manifest', {}).get('deployment', {}).get('arms', {})
            version = units.get('consumer_version') or {}
            if version:
                declared = str(version.get('declared_version') or 'not supplied')
                if version.get('comparison_allowed') is not True:
                    verdict = 'Recorded units identified; robot comparison INCOMPLETE'
                    problem = (f'Recorded units: {label}. ' +
                               ('Supply the LeRobot version from your rollout environment to compare the robot settings.'
                                if not version.get('declared_version') else
                                f'A verified units comparison is not available for LeRobot {declared} and this robot.'))
                    if not version.get('declared_version'):
                        next_command = 'python -m pip show lerobot  # run in the rollout environment; then add --lerobot-version'
                elif version.get('declared_version'):
                    problem += ' Using the declared LeRobot ' + declared + ' settings.'
                if version.get('command_role') == 'collection_only' or version.get('conditional_command_role') == 'collection_only':
                    problem = 'lerobot-record collects data; it does not run a trained policy. ' + problem
            elif any(str(arm.get('origin', '')).startswith('the default of ') for arm in arms.values()):
                problem += ' The robot version was not supplied; this comparison uses a recorded source default.'
            if camera.get('requested'):
                labels = {'pass': 'consistent', 'error': 'mismatch (ERROR)',
                          'warning': 'review needed (WARNING)', 'unknown': 'incomplete'}
                verdict = ('Robot settings: units ' + labels.get(unit_state, 'incomplete')
                           + '; camera names ' + labels.get(camera.get('status'), 'incomplete'))
                camera_findings = camera.get('findings', [])
                camera_issue = next((f for f in camera_findings if f.get('status') in ('error', 'unknown', 'warning')), None)
                if camera_issue:
                    missing = [str(key).removeprefix('observation.images.') for key in camera.get('missing', [])]
                    declared_keys = camera.get('effective_declared_keys', [])
                    if missing and not declared_keys:
                        camera_problem = 'The dataset has cameras ' + ', '.join(missing[:4]) + ', but the robot supplies none.'
                    elif missing:
                        camera_problem = 'Missing robot camera names: ' + ', '.join(missing[:4]) + '.'
                    else:
                        camera_problem = 'The camera configuration needs review; inspect its missing or extra names.'
                    problem = camera_problem + ' ' + problem
                    if missing and unit_state != 'error' and version.get('comparison_allowed', True):
                        next_command = ('Configure the actual --robot.cameras streams named ' + ', '.join(missing[:4])
                                        + ', then rerun this check. Use your real device paths or indexes.')
                elif camera.get('status') == 'pass':
                    problem += ' Dataset and declared camera keys match.'
                else:
                    problem += ' Camera-key assessment: ' + camera.get('status', 'unknown').upper() + '; request its missing or bounded evidence.'
                problem += ' Camera size/rate, joint order and robot execution are not checked.'
            else:
                problem += ' Cameras, size/rate, joint order and robot execution are not checked.'
            if unit_state == 'error':
                corrected = _unit_correction_command(units, arguments)
                if corrected:
                    next_command = corrected
                    if camera.get('status') in ('error', 'warning') and camera.get('missing'):
                        names = ', '.join(str(key).removeprefix('observation.images.') for key in camera['missing'][:4])
                        next_command += '  # also configure actual cameras: ' + clean(names, 100)
    elif kind == 'recovery':
        if report.get('kind') == 'recovery_copy':
            created = report.get('copy_created') is True
            unchanged = report.get('original_unchanged') is True
            verdict = ('Recovery copy: ' + state.upper() +
                       ('; original unchanged' if unchanged else '; original stability not established'))
            statistics = report.get('statistics') or {}
            stats_status = str(statistics.get('status', 'unknown')).upper()
            integrity = (report.get('verification') or {}).get('recording_integrity', 'unknown')
            if isinstance(integrity, dict):
                integrity = integrity.get('status', 'unknown')
            problem = ('New copy created. ' if created else 'No qualified recovery copy was created. ')
            problem += f'Recording integrity: {str(integrity).upper()}; supported statistics: {stats_status}. '
            if statistics.get('status') != 'pass':
                problem += 'Saved statistics were not established for the retained episodes; do not treat this copy as training/resume-ready. '
            elif statistics.get('method') == 'source_row_and_decoded_frame_regeneration':
                problem += 'Numeric and decoded RGB summaries were regenerated. '
            else:
                problem += 'Numeric values were compared; retained image summaries were not decoded. '
            problem += 'Native reload, actual resume and training remain separate checks.'
            destination = report.get('destination')
            if created and destination:
                next_command = shlex.join(['roboclue', 'check', str(destination), '--question', 'recovery'])
            return verdict, clean(problem, 620), next_command
        current = (report.get('plan') or {}).get('validity', {}).get('current_directory_resume_status', 'unknown')
        verdict = 'Recording integrity: ' + state.upper()
        if current == 'unsafe':
            verdict += '; this directory has a detected resume hazard'
        elif state != 'pass':
            verdict += '; required evidence remains incomplete'
        if state == 'pass':
            verdict = 'Recording integrity: covered checks PASS; native reload/resume still needs verification'
        episodes = report.get('episodes', {})
        kept = _ranges(episodes.get('keep')) or 'none verified'
        candidates = _ranges(episodes.get('recovery_candidates'))
        error = next((f for f in findings if f.get('code') == 'salvage.resume_hazard'), None)
        error = error or next((f for f in findings if f.get('code') == 'salvage.file_unreadable'), first)
        cause = clean(error.get('message', message), 230)
        if error.get('code') == 'salvage.file_unreadable':
            cause = 'Incomplete Parquet data was found; do not resume this directory as-is.'
        elif error.get('code') == 'salvage.resume_hazard':
            cause = 'Do not resume this directory: ' + clean(error.get('message', message), 180)
        if report.get('assessment_kind') == 'metadata_totals_mismatch':
            cause = 'Saved totals disagree with the complete recording ledger; a crashed recording was not established.'
        problem = f'{cause} A new metadata-only copy can keep episodes {kept}.'
        if candidates:
            problem += f' Extra candidate episodes {candidates} require decoding and metadata regeneration; preserve their files.'
        problem += ' This check did not modify files.'
        if state == 'pass':
            from .console_text import statistics_scope

            problem = (statistics_scope(report.get('statistics_comparison', {})) +
                       ' Selected recording rows, metadata and requested headers cleared; native reload remains separate.')
        if (report.get('plan') or {}).get('stats_note'):
            problem = ('meta/stats.json can still include dropped episodes; rebuild statistics for the retained episodes before training. '
                       + problem)
        if any(f.get('code') == 'salvage.video_unchecked' for f in findings):
            next_command = _rerun([a for a in arguments if a != '--no-video-headers'], '--video-headers')
            problem = 'Video evidence was not fully checked; enable headers or restore the cited local clips. ' + problem
        if report.get('assessment_kind') == 'metadata_rows_mismatch':
            verdict = 'Episode metadata: ERROR; identity columns support a repair'
            if current == 'unsafe':
                verdict += '; do not resume this directory'
            problem = (f'Row indices, episode labels and frame order support a metadata repair for episodes {kept}. '
                       'Other row values and native loading remain separate checks. This check did not modify files.')
            if any(f.get('code') == 'salvage.video_unchecked' for f in findings):
                problem = 'Video evidence is incomplete. ' + problem
            else:
                next_command = _rerun(arguments, '--explain')
        elif report.get('assessment_kind') == 'metadata_rows_unresolved':
            verdict = 'Episode metadata: REVIEW REQUIRED; do not resume this directory'
            if state in ('error', 'warning'):
                verdict += '; ' + state.upper() + ' in other checked evidence'
            problem = ('No keep/drop plan is approved. Metadata, row positions or stored indices disagree, '
                       'and a complete repair is not established. Preserve every original file; review the cited conflicts.')
            if any(f.get('code') == 'salvage.video_unchecked' for f in findings):
                problem = 'Video evidence was not fully checked. ' + problem
            else:
                next_command = _rerun(arguments, '--explain')
    elif kind == 'sim':
        pairs = [f for f in findings if f.get('code') == 'isaaclab.conversion_quaternion_order']
        pair_pass = pairs and all(f.get('status') == 'pass' for f in pairs)
        if pair_pass and state not in ('error',):
            verdict = 'Declared quaternion conversion: CONSISTENT' + ('; other checks remain incomplete' if state == 'unknown' else '')
            problem = 'The paired source/export declarations and quaternion transformation agree. Content-based order confirmation, other conventions and physical behavior are separate checks.'
        else:
            verdict = 'Simulator-data conventions: ' + state.upper()
            problem = message
            if first.get('code') == 'isaaclab.format_version' and 'No root format_version' in message:
                problem = ('This file has no root format_version. Isaac Lab 3.0 interprets it as legacy WXYZ; '
                           'the numeric values alone have not established the intended order. Compare a declared source/export pair before changing data.')
    elif kind == 'groot':
        versions = report.get('versions', {})
        label = '; '.join(f"{name.upper()}: {part.get('status', 'unknown').upper()}" for name, part in versions.items())
        if not label:
            label = str(report.get('groot_version') or 'version not established').upper() + ': ' + state.upper()
        verdict = 'GR00T metadata/config — ' + label
        if report.get('selected_assessment', {}).get('name') == 'groot_consumer_selection':
            return ('Choose the GR00T consumer version; ' + label,
                    'The version-specific results differ. Choose the version you actually train with; no version was assumed.',
                    'Add --groot-version n1.5 or --groot-version n1.7 to this check, matching your training environment.')
        if state in ('pass', 'warning'):
            problem = 'No error in the selected GR00T metadata/config contract. Native rows, decoded cameras, model loading and training remain unassessed.'
            if state == 'warning':
                problem += ' Review: ' + message
        else:
            problem = str(report.get('headline') or message) + ' ' + message
        if report.get('mode') == 'both':
            problem = 'No consumer version was selected; these are separate profile results. Use --groot-version n1.5 or n1.7 for your actual consumer. ' + problem
        if any(f.get('code') in ('groot.language_key', 'groot.config_key', 'groot.video_key') and f.get('status') == 'error' for f in findings):
            problem += ' Check that the selected training config matches this dataset before considering any data edit.'
    elif kind == 'conventions':
        part = report.get('components', {}).get('action_conventions', {})
        context = report.get('components', {}).get('openpi_context', {})
        verdicts = part.get('verdicts', {})
        consistent = verdicts.get('consumer') == 'pass'
        verdict = 'Delta-transform choice: ' + ('CONSISTENT' if consistent else 'INCOMPLETE/REVIEW')
        severity = [f for f in part.get('findings', []) if f.get('status') in ('error', 'warning')]
        severity.sort(key=lambda f: 0 if f['status'] == 'error' else 1)
        if severity:
            level = 'ERROR' if any(f['status'] == 'error' for f in severity) else 'WARNING'
            verdict += '; convention ' + level
            names = ', '.join(dict.fromkeys(f.get('code', 'unnamed finding') for f in severity))
            cause = str(severity[0].get('message') or severity[0].get('reason') or 'Review the cited convention evidence').rstrip('.')
            problem = f"{names}: {cause}. "
        elif part.get('status') in ('warning', 'error'):
            verdict += '; convention ' + part['status'].upper() + ' (finding examples omitted)'
            problem = 'The convention component reports ' + part['status'].upper() + '; inspect its bounded findings with --explain. '
        else:
            problem = 'The declared delta-transform choice was assessed. '
        unresolved = [str(f.get('message') or f.get('code')) for f in part.get('findings', []) if f.get('status') == 'unknown']
        if unresolved:
            problem += 'Unresolved: ' + unresolved[0] + ' '
        if context:
            context_state = context.get('status', 'unknown')
            verdict = ('OpenPI input keys: ' + {'pass': 'CONSISTENT', 'error': 'MISMATCH',
                                               'unknown': 'INCOMPLETE'}.get(context_state, 'REVIEW') + '; ' + verdict)
            if context_state == 'error' and context.get('missing_keys'):
                keys = ', '.join(str(value) for value in context['missing_keys'][:4])
                more = ' (more in --explain)' if len(context['missing_keys']) > 4 else ''
                problem = f'The selected OpenPI adapter needs missing raw keys: {keys}{more}. Choose an adapter matching the dataset. '
                next_command = 'Choose or supply a matching OpenPI repack adapter, then rerun this check with its --openpi-config.'
            elif context_state == 'unknown':
                problem = 'The selected OpenPI raw-input contract was not established. ' + problem
            dataset_format = context.get('dataset_codebase_version')
            if dataset_format:
                problem += f'Dataset format: {dataset_format}. '
            problem += 'The consumer lockfile pin is reported in --explain; installed-loader compatibility and training remain unverified.'
        else:
            problem += 'Repack keys, loader versions, image mapping and native training remain unassessed.'
        if report.get('status') == 'error':
            verdict += '; other check errors remain'
    else:
        return None
    return verdict, clean(problem, 620), next_command
