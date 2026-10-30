"""Small, scoped deployment-unit answers for the public MCP entry point.

This is a presentation of the original preflight report, never a new detector.
Legacy and advanced transports retain their existing report shape and bounds.
"""
from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

BRIEF_LIMIT = 2048
BRIEF_RESULT_LIMIT = 2560
UNASSESSED_INTERFACES = ['image size', 'frame rate/fps', 'joint order']


def attach_consumer_version(report: dict, declared_version: str | None, *, require_version=False) -> dict:
    """Replace only the active units comparison; other errors remain errors."""
    from .consumer_version import apply_consumer_version, consumer_version_context
    from .preflight import _status

    result = dict(report)
    units = (report.get('components') or {}).get('units')
    if not isinstance(units, dict):
        result['consumer_version'] = consumer_version_context(declared_version)
        return result
    scoped = apply_consumer_version(units, declared_version, require_version=require_version)
    result['components'] = dict(report['components'], units=scoped)
    result['consumer_version'] = scoped['consumer_version']
    result['findings'] = [finding for finding in report.get('findings', []) if finding.get('component') != 'units']
    result['findings'].extend(dict(finding, component='units') for finding in scoped.get('findings', []))
    result['status'] = _status([part.get('status', 'unknown') for part in result['components'].values()
                                if isinstance(part, dict)])
    result['counts'] = {state: sum(finding.get('status') == state for finding in result['findings'])
                        for state in ('error', 'warning', 'unknown', 'pass')}
    return result


def attach_camera_comparison(report: dict, camera: dict) -> dict:
    """Join an explicit keys-only check without altering the engine report."""
    from .preflight import _status

    result = dict(report)
    result['interface_scope'] = {'camera_keys': camera.get('status') if camera.get('compared') else 'not_assessed',
                                 **{name: 'not_assessed' for name in UNASSESSED_INTERFACES}}
    result['coverage'] = dict(report.get('coverage') or {}, camera_keys_requested=bool(camera.get('requested')))
    if camera.get('requested'):
        result['components'] = dict(report.get('components') or {}, camera_keys=camera)
        result['findings'] = [*report.get('findings', []),
                              *[dict(finding, component='camera_keys') for finding in camera.get('findings', [])]]
        result['status'] = _status([report.get('status', 'unknown'), camera.get('status', 'unknown')])
        result['counts'] = {state: sum(f.get('status') == state for f in result['findings'])
                            for state in ('error', 'warning', 'unknown', 'pass')}
    return result


def _camera_view(camera: dict | None) -> dict | None:
    from .rule_catalog import rules_for_code

    if not isinstance(camera, dict):
        return None
    prefix = 'observation.images.'
    result = {'status': camera.get('status', 'unknown'), 'compared': bool(camera.get('compared')),
              'feature_prefix': prefix}
    if (camera.get('coverage') or {}).get('default_camera_declaration'):
        result['defaulted'] = True
    for output, key in (('dataset', 'dataset_keys'), ('declared', 'effective_declared_keys'),
                        ('missing', 'missing'), ('extra', 'extra')):
        values = camera.get(key) or []
        if values or output in ('dataset', 'declared'):
            result[output] = [value.removeprefix(prefix) for value in values]
    raw_names = camera.get('declared_keys') or []
    if raw_names != (camera.get('effective_declared_keys') or []):
        result['declared_raw'] = [value.removeprefix(prefix) for value in raw_names]
    findings = camera.get('findings') or []
    if findings:
        result['code'] = findings[0].get('code')
        rule_ids = [rule['id'] for rule in rules_for_code(result['code'])] if result['code'] else []
        if rule_ids:
            result['rule_ids'] = rule_ids
    for finding in findings:
        for reference in finding.get('evidence') or []:
            if reference.get('pointer') == '/features' and reference.get('sha256'):
                result['metadata_sha256'] = reference['sha256']
    source = camera.get('robot_source') or {}
    if source.get('sha256'):
        result['declaration_sha256'] = source['sha256']
    if len(json.dumps(result).encode()) > 512:
        return {'status': camera.get('status', 'unknown'), 'compared': bool(camera.get('compared')),
                'keys_require_details': True}
    return result


def _fits(value: dict) -> bool:
    return len(json.dumps(value, allow_nan=False).encode()) <= BRIEF_LIMIT


def _reference(item: dict, source: str) -> dict:
    path = item.get('path')
    if isinstance(path, str):
        try:
            relative = Path(path).relative_to(source)
            path = '$dataset' + ('/' + relative.as_posix() if relative.parts else '')
        except ValueError:
            pass
    return {key: value for key, value in (
        ('path', path), ('pointer', item.get('pointer')), ('sha256', item.get('sha256')))
        if value is not None}


def deployment_answer(report: dict, *, source: str, robot_command: str | None = None,
                      robot_config: str | None = None) -> dict:
    """Keep the original overall verdict and explicitly answer units only.

    Full conditions are never cut off to meet the brief bound. If a witness or
    command cannot fit, return an explicit unresolved view and require details.
    The original ERROR remains ERROR even when its detailed witness is withheld.
    """
    from .framework_versions import collect_framework_versions, compact_framework_context
    from .product_questions import _unit_correction_command, units_view
    from .rule_catalog import rules_for_code

    digest = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    observed_framework = compact_framework_context(collect_framework_versions(), 'lerobot')
    framework = {'environment': 'Tool Python; consumer environment unverified.',
                 'runtime_revalidated': False}
    if observed_framework.get('version'):
        framework['version'] = observed_framework['version']
    units = report.get('components', {}).get('units')
    camera = report.get('components', {}).get('camera_keys')
    unresolved = []
    if report.get('status') == 'unknown' and isinstance(units, dict) and units.get('status') == 'pass':
        labels = {'episode_windows': 'episode/video evidence', 'video_index': 'episode/video evidence',
                  'robot_readiness': 'policy interface', 'dataset_statistics': 'saved statistics'}
        unresolved = sorted({labels.get(name, name) for name, part in report.get('components', {}).items()
                             if name not in ('units', 'camera_keys') and isinstance(part, dict) and part.get('status') == 'unknown'})
    unassessed = [*UNASSESSED_INTERFACES]
    if not isinstance(camera, dict) or not camera.get('compared'):
        unassessed.insert(0, 'camera keys')
    detail = "Repeat this request with response_format='detailed' (new read)."
    context = report.get('consumer_version') or (units or {}).get('consumer_version')
    consumer = None
    if isinstance(context, dict):
        consumer = {'declared': context.get('declared_version'), 'profile': context.get('tested_profile'),
                    'role': context.get('command_role'), 'fleet_verified': False}
        if not context.get('comparison_allowed'):
            consumer['reason'] = context.get('reason', 'Consumer comparison has no qualified profile.')
            if context.get('conditional_command_role') == 'collection_only':
                consumer['conditional_role'] = 'collection_only'
        detail = 'response_format=detailed; new read'
    if not isinstance(units, dict):
        missing = {'kind': 'deployment_units_answer',
                'status': 'error' if report.get('status') == 'error' else 'unknown',
                'report_status': report.get('status', 'unknown'),
                'units_status': 'unknown', 'deployment_comparison': 'not_assessed',
                'problem': report.get('coverage', {}).get('units_not_run', 'Requested units check is absent.'),
                'scope': 'No deployment approval; the required recorded-units comparison did not run.',
                'not_assessed': unassessed, 'framework_context': framework,
                'full_report_sha256': digest, 'details': detail}
        if camera is not None:
            missing['camera_keys'] = _camera_view(camera)
        if consumer is not None:
            missing['consumer'] = consumer
        if not _fits(missing):
            missing['problem'] = 'Requested units check is absent; complete reason requires details.'
        return missing
    manifest = units.get('manifest') or {}
    view = units_view(units)
    # Camera names have their own code and metadata/command witnesses below.
    # Keep the numerical units witness when that component makes a positive claim.
    findings = sorted(units.get('findings') or (camera.get('findings') or [] if isinstance(camera, dict) else []),
                      key=lambda f: {'error': 0, 'warning': 1, 'unknown': 2}.get(f.get('status'), 3))
    first = findings[0] if findings else {}
    command = ['roboclue', 'check', source]
    if robot_command is not None:
        command += ['--robot-command', robot_command]
    if robot_config is not None:
        command += ['--robot-config', robot_config]
    corrected = _unit_correction_command(units, command) if units.get('status') == 'error' else None
    configured = sorted({part['mode'] for part in (manifest.get('deployment') or {}).get('arms', {}).values()
                         if part.get('mode')})
    part = {'status': units.get('status', 'unknown'), 'compared_with': units.get('compared_with'),
            'detected_units': sorted({mode for modes in view['detected_units'].values() for mode in modes}),
            'declared_units': configured}
    # The pinned-default and inference caveats are not interchangeable with a
    # generic disclaimer: preserve the detector's complete units limits.
    part['limits'] = units.get('units_limits') or view['scope']
    if context:
        part['limits'] = ('Units inferred; policy outputs assumed unchanged. Consumer defaults use the declared source profile; processors, calibration and robot uninspected.'
                          if context.get('comparison_allowed') else
                          'Recorded units only. Robot units and defaults are not established for this version; policy processors, calibration and robot were not inspected.')
    result = {'kind': 'deployment_units_answer', 'status': report.get('status', 'unknown'),
              'scope': 'Units/settings only; no deployment approval.',
              'framework_context': framework,
              'components': {'units': part},
              'not_assessed': unassessed,
              'finding': {'code': first.get('code'), 'status': first.get('status'),
                          'rule_ids': [rule['id'] for rule in rules_for_code(first['code'])] if first.get('code') else [],
                          'evidence': [_reference(item, source) for item in first.get('evidence', [])]},
              'next': ("Repeat preflight_check with the same source and robot_command=" +
                       json.dumps(shlex.split(corrected)[-1])) if corrected else detail,
              'full_report_sha256': digest, 'details': detail}
    if camera is not None:
        result['camera_keys'] = _camera_view(camera)
        if consumer is not None and result['camera_keys'].get('status') == 'pass':
            # The same exact code remains in the original detailed report and
            # RC007 mapping. Reserve brief space for the explicitly selected
            # consumer profile; retain every key, digest and comparison status.
            result['camera_keys'].pop('code', None)
        if camera.get('status') in ('error', 'warning', 'unknown'):
            result['next'] = 'Review camera names/rename_map, then repeat preflight_check.'
    if consumer is not None and corrected is not None and not (camera or {}).get('requested'):
        result['next'] = 'Repeat the same preflight_check request; replace only robot_command=' + json.dumps(shlex.split(corrected)[-1])
    elif corrected is not None and (camera or {}).get('status') == 'pass':
        result['next'] = 'Repeat the same preflight_check request; replace only robot_command=' + json.dumps(shlex.split(corrected)[-1])
    if camera and camera.get('status') in ('error', 'warning', 'unknown'):
        unit_fix = ('Review ' + ('--robot.use_degrees=false' if '--robot.use_degrees=false' in corrected else '--robot.use_degrees=true') + '; ') if corrected else ''
        result['next'] = unit_fix + 'provide actual --robot.cameras/rename_map; repeat this request.'
    if context and not context.get('comparison_allowed'):
        version_step = ('Supply lerobot_version from the actual robot environment.' if not context.get('declared_version')
                        else 'Use a supported source profile for the actual robot environment; do not relabel its version to pass.')
        result['next'] = version_step + (' ' + result['next'] if camera and camera.get('status') in ('error', 'warning') else '')
    if result.get('next') == detail:
        result.pop('next')
    if unresolved:
        result['unresolved'] = unresolved
    if consumer is not None:
        result['consumer'] = consumer
        if first.get('status') == 'pass' and part['status'] == 'pass' and str(first.get('code', '')).startswith('units.'):
            # This repeats the complete units component verdict. Keep the code,
            # all witnesses and limits, reserving space for the fleet profile.
            # Non-pass finding severities are never removed here.
            result['finding'].pop('status', None)
    if _fits(result) and not (result.get('camera_keys') or {}).get('keys_require_details'):
        return result
    # Do not silently truncate a long source/command/witness or retain a positive
    # concise conclusion after its conditions were omitted. Details remain an
    # actual read of the same inputs, not a cached report or authority grant.
    withheld = {'kind': 'deployment_units_answer',
            'status': 'error' if report.get('status') == 'error' else 'unknown',
            'report_status': report.get('status', 'unknown'), 'units_status': units.get('status', 'unknown'),
            'answer_status': 'details_required',
            'scope': 'No concise approval. Comparison witnesses/settings require the detailed view.',
            'not_assessed': unassessed, 'framework_context': framework,
            'full_report_sha256': digest, 'details': detail}
    if camera is not None:
        withheld['camera_keys'] = _camera_view(camera)
    if unresolved:
        withheld['unresolved'] = unresolved
    if consumer is not None:
        withheld['consumer'] = consumer
    return withheld


def human_unit_mode(mode: str) -> str:
    """Readable mode names; exact detector values stay in structured evidence."""
    return {'RANGE_M100_100': '-100..100 across the calibrated range',
            'RANGE_0_100': '0..100 gripper scale', 'DEGREES': 'degrees',
            'RADIANS': 'radians', 'NOT_M100': 'not the -100..100 scale'}.get(mode, str(mode))


def _deployment_text(answer: dict) -> str:
    units = answer.get('components', {}).get('units', {})
    status = answer.get('status', 'unknown')
    title = {'pass': 'Covered settings consistent', 'warning': 'Review needed',
             'error': 'Mismatch found', 'unknown': 'Check incomplete'}.get(status, status)
    lines = [f"{title} ({status.upper()})."]
    if answer.get('answer_status') == 'details_required':
        lines.append('The full settings and evidence do not fit this brief answer. Request response_format=\'detailed\'.')
    elif units:
        recorded = '; '.join(human_unit_mode(value) for value in units.get('detected_units', [])) or 'not determined'
        declared = '; '.join(human_unit_mode(value) for value in units.get('declared_units', [])) or 'not established'
        if len(units.get('declared_units', [])) == 1 and set(units['declared_units']) <= set(units.get('detected_units', [])) <= set(units['declared_units']) | {'RANGE_0_100'}:
            declared = 'same joint scale'
        lines.append(f"Units {units.get('status', 'unknown').upper()}: recorded {recorded}; robot {declared}.")
    else:
        lines.append(str(answer.get('problem') or 'The requested units comparison did not run.'))
    camera = answer.get('camera_keys')
    if camera is not None:
        names = ', '.join(camera.get('dataset') or []) or 'none'
        declared = ', '.join(camera.get('declared') or []) or ('none' if camera.get('compared') else 'not established')
        lines.append(f"Camera keys {camera.get('status', 'unknown').upper()}: dataset {names}; configured {declared}.")
    consumer = answer.get('consumer')
    if consumer:
        version = consumer.get('declared')
        if not version:
            lines.append('Robot version not supplied; provide lerobot_version from the actual robot environment.')
        elif consumer.get('profile') is None:
            lines.append(f'No verified units profile for the declared LeRobot {version} on this robot.')
        else:
            note = '; collection settings only' if consumer.get('role') == 'collection_only' else ''
            lines.append(f'LeRobot {version} declared{note}; robot not inspected.')
        if consumer.get('conditional_role') == 'collection_only':
            lines.append('In tested sources, this record command collects data only.')
    followup = answer.get('next') or ''
    hint = '--robot.use_degrees=false' if '--robot.use_degrees=false' in followup else (
        '--robot.use_degrees=true' if '--robot.use_degrees=true' in followup else None)
    if camera and camera.get('status') in ('error', 'warning', 'unknown'):
        wanted = ', '.join(camera.get('dataset') or [])
        lines.append('Next: ' + (f'Review {hint}; ' if hint else '') +
                     'supply actual --robot.cameras' + (f' for {wanted}' if wanted else '') + '; recheck.')
    elif hint:
        lines.append(f'Next: review {hint}, then repeat the same check. Exact command is in structuredContent.')
    code = answer.get('finding', {}).get('code')
    if code and answer.get('finding', {}).get('status') in ('error', 'warning', 'unknown'):
        rules = answer.get('finding', {}).get('rule_ids') or []
        lines.append('Evidence: ' + code + (' (' + ', '.join(rules) + ')' if rules else '') + '.')
    if answer.get('unresolved'):
        lines.append('Other evidence missing; see structuredContent.')
    lines.append('Unassessed: image size, frame rate/fps, joint order, policy processing, physical behavior.')
    return '\n'.join(lines)


def deployment_tool_result(answer: dict):
    """Readable text plus one structured evidence view, with the original bounds."""
    from mcp.types import CallToolResult, TextContent

    result = CallToolResult(content=[TextContent(type='text', text=_deployment_text(answer))], structuredContent=answer)
    if len(json.dumps(result.model_dump(mode='json', exclude_none=True)).encode()) <= BRIEF_RESULT_LIMIT:
        return result
    units = answer.get('components', {}).get('units', {})
    withheld = {'kind': 'deployment_units_answer',
                'status': 'error' if answer['status'] == 'error' else 'unknown',
                'report_status': answer.get('report_status', answer['status']),
                'units_status': units.get('status', answer.get('units_status', 'unknown')),
                'answer_status': 'details_required',
                'scope': 'No concise approval. Complete settings/evidence exceed the brief result bound.',
                'not_assessed': answer.get('not_assessed', UNASSESSED_INTERFACES),
                'framework_context': answer.get('framework_context'),
                'full_report_sha256': answer['full_report_sha256'], 'details': answer['details']}
    for key in ('camera_keys', 'unresolved', 'consumer'):
        if answer.get(key) is not None:
            withheld[key] = answer[key]
    return CallToolResult(content=[TextContent(type='text', text=_deployment_text(withheld))], structuredContent=withheld)
