"""Bounded, read-only run discovery and triage using existing diagnostic engines."""
from __future__ import annotations

import json
import math
import os
import re
import stat
from copy import deepcopy
from pathlib import Path

from .artifacts import InputError, read_json, read_json_value
from .capture import audit_capture
from .diagnostics import audit
from .rollouts import inspect_rollout
from .trajectory import _load as load_trajectory
from .trajectory import _read_document as read_trajectory_document

SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache'}


def _discover_case_boundary(directory, inventory, coverage, problems, max_files, manifest_cache):
    """Keep an unverified case envelope opaque before inspecting its inputs."""
    manifest, ready = directory / 'manifest.json', directory / 'READY.json'
    try:
        manifest_stat = manifest.lstat()
    except FileNotFoundError:
        manifest_stat = None
    except OSError as exc:
        problems.append({'path': str(directory), 'reason': str(exc)})
        return True
    try:
        ready_stat = ready.lstat()
    except FileNotFoundError:
        ready_stat = None
    except OSError as exc:
        problems.append({'path': str(directory), 'reason': str(exc)})
        return True
    if manifest_stat is None and ready_stat is None:
        return False
    if manifest_stat is None and stat.S_ISREG(ready_stat.st_mode) and ready_stat.st_size <= 1024 * 1024:
        # Native producers also publish READY, naming capture.json/selection.json
        # instead of manifest.json. Keep their existing explicit audit route;
        # a copied checkpoint/dataset inside an incomplete case is not a capture.
        for name in ('capture.json', 'selection.json'):
            native = directory / name
            try:
                native_stat = native.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(native_stat.st_mode) or native_stat.st_size > 1024 * 1024:
                continue
            if coverage['files_seen'] + 1 > max_files or coverage['bytes_inspected'] + native_stat.st_size > 16 * 1024 * 1024:
                coverage['truncated'] = True
                return True
            coverage['files_seen'] += 1
            coverage['bytes_inspected'] += native_stat.st_size
            try:
                native_data = read_json(native)
            except InputError:
                continue
            manifest_cache[str(native)] = (native_data, native_stat.st_size)
            kind = native_data.get('kind')
            recognized = (name == 'selection.json' and kind == 'act_selection_capture') or (
                name == 'capture.json' and (kind in ('named_action_capture', 'command_path_capture', 'resolved_layout_capture') or
                (type(native_data.get('has_deployment')) is bool and isinstance(native_data.get('files'), list))))
            if recognized:
                return False
    entries = tuple(entry for entry in (manifest_stat, ready_stat) if entry is not None)
    if any(stat.S_ISLNK(entry.st_mode) for entry in entries):
        coverage['symlinks_skipped'] += sum(stat.S_ISLNK(entry.st_mode) for entry in entries)
        problems.append({'path': str(directory), 'reason': 'Case envelope symlink refused; internal inputs were not inspected.'})
        return True
    if not all(stat.S_ISREG(entry.st_mode) for entry in entries):
        problems.append({'path': str(directory), 'reason': 'Case envelope must use regular manifest/READY files; internals were not inspected.'})
        return True
    if coverage['files_seen'] + len(entries) > max_files:
        coverage['truncated'] = True
        return True
    coverage['files_seen'] += len(entries)
    manifest_size = manifest_stat.st_size if manifest_stat is not None else 0
    if any(entry.st_size > 1024 * 1024 for entry in entries) or coverage['bytes_inspected'] + manifest_size > 16 * 1024 * 1024:
        coverage['truncated'] = True
        problems.append({'path': str(directory), 'reason': 'Case envelope exceeds discovery budget; internal inputs were not inspected.'})
        return True
    if manifest_stat is None:
        inventory['incomplete_cases'].append(str(directory))
        problems.append({'path': str(manifest), 'reason': 'Case manifest is missing while READY exists; internal inputs were not inspected. Explicit replay must verify/refuse this incomplete case.'})
        return True
    coverage['bytes_inspected'] += manifest_size
    try:
        data = read_json(manifest)
    except InputError as exc:
        problems.append({'path': str(manifest), 'reason': str(exc)})
        return True
    key = {'robotics_retained_check': 'retained_checks', 'robot_video_reference_recovery': 'retained_checks', 'robot_incident_case': 'incident_cases',
           'trajectory_regression_case': 'trajectory_cases', 'native_trajectory_sample_case': 'trajectory_cases'}.get(
               data.get('kind') if isinstance(data, dict) and isinstance(data.get('kind'), str) else None)
    if key is None:
        if ready_stat is None:
            # A normal run can have an unrelated manifest. Reuse the bounded
            # read in the ordinary classifier rather than reading it twice.
            manifest_cache[str(manifest)] = (data, manifest_stat.st_size)
            return False
        problems.append({'path': str(manifest), 'reason': 'Unrecognized case envelope; internal inputs were not inspected.'})
    else:
        inventory[key].append(str(directory))
        if ready_stat is None:
            problems.append({'path': str(ready), 'reason': 'Case commit marker is missing; explicit replay must verify/refuse this incomplete case.'})
    return True


def _trajectory_candidate(path):
    names, times, points, _, artifact = load_trajectory(path)
    # Bounded suggestions expose existing sampling; they do not select a live controller model.
    suggested = set()
    for left, right in zip(times[:9], times[1:9], strict=False):
        suggested.update(left + (right - left) * part // 4 for part in range(5))
    if len(times) == 1:
        suggested.add(times[0])
    return {**artifact, 'point_count': len(points), 'joint_names': names,
            'waypoint_time_range_ns': [times[0], times[-1]],
            'derivative_points': {key: sum(bool(len(p[key])) for p in points)
                                  for key in ('velocities', 'accelerations', 'effort')},
            'suggested_sample_times': [ns / 1e9 for ns in sorted(suggested) if ns <= 3600 * 10**9],
            'suggestion_scope': 'Quarter-segment samples in the first eight segments only; not an extrema search or whole-path qualification.',
            'interpolation_scope': 'Explicit pinned splines reference model; live controller parameters, positions upsampling, initial state and replacement behavior are unverified.',
            'execution_status': 'unknown'}


def _capture_summary(report):
    """Keep engine-specific scopes; a consumer verdict is not a generic finding table."""
    keys = ('kind', 'status', 'counts', 'artifacts', 'scope', 'binding_status', 'numeric_replay_status',
            'acceptance_contract', 'action_names', 'engine_action_names', 'driver_namespace',
            'tick_count', 'driver_change_events', 'canonical_mutation_events', 'first_driver_change',
            'recorded_action_status', 'recording', 'physical_execution_status', 'policy_selection_status',
            'policy_processor_status', 'robot_processor_semantics_status', 'raw_binding', 'processed_binding',
            'layout', 'unverified', 'cadence_criterion', 'consumer_requirement', 'query_lineage_status',
            'downstream_association', 'observed_processor', 'consumer_namespace', 'tolerances')
    return {key: report[key] for key in keys if key in report}


def _capture_findings(report):
    if isinstance(report.get('findings'), list):
        return [finding for finding in report['findings'] if finding.get('status') != 'pass']
    findings = []
    contract = report.get('acceptance_contract', {})
    for violation in contract.get('violations', [])[:16]:
        findings.append({'code': report.get('kind', 'capture') + '.consumer_criterion', 'status': 'error',
                         'message': 'Explicit consumer requirement differs: ' + str(violation.get('field', 'criterion')),
                         'evidence': violation.get('evidence', [])})
    intrinsic = report.get('first_intrinsic_contradiction') or report.get('first_numeric_contradiction')
    if report.get('numeric_replay_status') == 'error' and intrinsic:
        findings.append({'code': report.get('kind', 'capture') + '.numeric_contradiction', 'status': 'error',
                         'message': 'Supported native arithmetic contradicts captured evidence.',
                         'evidence': intrinsic.get('evidence', [])})
    return findings


def _preview(value):
    if isinstance(value, dict):
        return dict(list(value.items())[:8]), len(value), len(value) > 8
    if isinstance(value, list):
        return value[:8], len(value), len(value) > 8
    return value, 0, False


def _boundary_comparison(reference, candidate, *, atol=1e-6, rtol=1e-5):
    """Compare explicit native send ordinals, never infer synchronization or causation."""
    from .named_actions import DATASET_FILES, _load, _parquet_recording, _recheck, audit_named_actions

    report = {'kind': 'run_boundary_comparison', 'status': 'unknown', 'comparison_supported': False,
              'difference_is_fault': False, 'equal_over_compared_ticks': None,
              'alignment': 'Explicit reference/candidate ordinal native send calls only; not time, episode or physical alignment.',
              'first_supported_difference': None, 'context_differences': [], 'earlier_unknown_stages': [],
              'compared_tick_count': 0, 'recording_comparison': {'comparison_supported': False, 'equal': None, 'first_difference': None},
              'unknown_stages': ['interpolated', 'canonical_input', 'robot_processor_output', 'driver_request', 'driver_return', 'canonical_return', 'native_writer_row'],
              'whole_path_equivalence': 'unknown', 'earlier_unknown_stage_count': 0, 'compared_stages': [],
              'scope': 'Observed software boundary differences only. First supported difference is not a causal diagnosis or fault; only an explicit criterion establishes incompatibility. Wrapper request/return bridges are audited separately and are not separate compared stages.'}
    report['tolerances'] = {'atol': atol, 'rtol': rtol}
    audits = [audit_named_actions(str(path), atol=atol, rtol=rtol) for path in (reference, candidate)]
    loaded = [_load(str(path)) for path in (reference, candidate)]
    try:
        for audit, (root, _, hashes, _) in zip(audits, loaded, strict=True):
            evidence = next(item for item in audit['artifacts'] if item['path'] == str(root / 'capture.json'))
            if evidence['sha256'] != hashes['capture.json']:
                raise InputError('Capture changed between audit and boundary comparison')
        if any(audit['binding_status'] != 'pass' or audit['numeric_replay_status'] != 'pass' for audit in audits):
            report['reason'] = 'A complete supported named boundary is missing or inconsistent; use the individual audit evidence.'
            return report
        left, right = [item[1] for item in loaded]
        for key in ('action_names', 'engine_action_names', 'driver_namespace'):
            if left[key] != right[key]:
                a, ac, at = _preview(left[key])
                b, bc, bt = _preview(right[key])
                report['context_differences'].append({'field': key, 'reference': a, 'candidate': b,
                    'reference_value_count': ac, 'candidate_value_count': bc, 'values_truncated': at or bt})
        report['reference_tick_count'], report['candidate_tick_count'] = len(left['ticks']), len(right['ticks'])
        report['compared_tick_count'] = min(len(left['ticks']), len(right['ticks']))
        stages = (('interpolated', 'interpolated'), ('canonical_input', 'canonical_at_processor'),
                  ('robot_processor_output', 'robot_processor_output'), ('driver_request', 'driver_request'),
                  ('driver_return', 'driver_return'), ('canonical_return', 'canonical_at_return'))
        stage_order = {name: index for index, (name, _) in enumerate(stages)}
        stage_order['native_writer_row'] = len(stages)
        report['compared_stages'] = [name for name, _ in stages]
        report['unknown_stages'] = ['native_writer_row']
        def evidence(side, pointer):
            root, _, hashes, _ = loaded[side]
            return {'path': str(root / 'capture.json'), 'sha256': hashes['capture.json'], 'pointer': pointer}
        def different(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                return set(a) != set(b) or any(not math.isclose(a[key], b[key], abs_tol=atol, rel_tol=rtol) for key in a)
            return a != b
        def difference(index, stage, a, b, pointers):
            changed = [key for key in dict.fromkeys([*a, *b]) if key not in a or key not in b
                       or not math.isclose(a[key], b[key], abs_tol=atol, rel_tol=rtol)]
            av, bv = {key: a.get(key) for key in changed[:8]}, {key: b.get(key) for key in changed[:8]}
            ac, bc, at, bt = len(a), len(b), len(changed) > 8, len(changed) > 8
            return {'tick_ordinal': index, 'stage': stage, 'reference': av, 'candidate': bv,
                    'changed_channel_count': len(changed), 'omitted_changed_channel_count': max(0, len(changed) - 8),
                    'reference_value_count': ac, 'candidate_value_count': bc, 'values_truncated': at or bt,
                    'evidence': [evidence(side, pointer) for side, pointer in enumerate(pointers)]}
        for index, (a, b) in enumerate(zip(left['ticks'], right['ticks'], strict=False)):
            for stage, field in stages:
                av, bv = a.get(field), b.get(field)
                if field == 'interpolated':
                    av = dict(zip(left['action_names'], av, strict=True)) if av is not None else None
                    bv = dict(zip(right['action_names'], bv, strict=True)) if bv is not None else None
                if av is None or bv is None:
                    report['earlier_unknown_stages'].append({'tick_ordinal': index, 'stage': stage})
                elif different(av, bv) and report['first_supported_difference'] is None:
                    report['first_supported_difference'] = difference(index, stage, av, bv,
                        [f'/ticks/{index}/{field}', f'/ticks/{index}/{field}'])
        recording = report['recording_comparison']
        if all(audit.get('recorded_action_status') == 'pass' for audit in audits):
            row_maps = [{row['tick_id']: row for row in item[1]['recording']['rows']} for item in loaded]
            if set(row_maps[0]) == set(row_maps[1]):
                recording.update(comparison_supported=True, equal=True)
                report['compared_stages'].append('native_writer_row')
                report['unknown_stages'] = []
                native_rows = [_parquet_recording(item[0], 'dataset/' + DATASET_FILES[0], ('action',), item[2],
                    rows=len(item[1]['recording']['rows']), width=len(item[1]['recording']['rows'][0]['action_names']))
                    for item in loaded]
                for tick_id in sorted(row_maps[0]):
                    rows = [mapping[tick_id] for mapping in row_maps]
                    values = [native_rows[side][row['index']]['action'] for side, row in enumerate(rows)]
                    av, bv = [dict(zip(row['action_names'], [value] if len(row['action_names']) == 1 else value, strict=True))
                              for row, value in zip(rows, values, strict=True)]
                    if different(av, bv):
                        delta = difference(tick_id, 'native_writer_row', av, bv,
                            [f"/recording/rows/{row['index']}/buffer_action" for row in rows])
                        delta['evidence'] = [{'path': str(loaded[side][0] / 'dataset' / DATASET_FILES[0]),
                            'sha256': loaded[side][2]['dataset/' + DATASET_FILES[0]], 'row': row['index'], 'column': 'action'}
                            for side, row in enumerate(rows)]
                        recording.update(equal=False, first_difference=delta)
                        first = report['first_supported_difference']
                        if first is None or (tick_id, stage_order['native_writer_row']) < (first['tick_ordinal'], stage_order[first['stage']]):
                            report['first_supported_difference'] = delta
                        break
        report['comparison_supported'] = True
        report['equal_over_compared_ticks'] = report['first_supported_difference'] is None
        report['complete_ordinal_coverage'] = len(left['ticks']) == len(right['ticks']) and not report['earlier_unknown_stages']
        unknowns = report['earlier_unknown_stages']
        first = report['first_supported_difference']
        if first is not None:
            unknowns = [item for item in unknowns if (item['tick_ordinal'], stage_order[item['stage']]) <
                        (first['tick_ordinal'], stage_order[first['stage']])]
        report['earlier_unknown_stage_count'] = len(unknowns)
        report['earlier_unknown_stages'] = unknowns[:12]
        return report
    finally:
        for root, _, hashes, _ in loaded:
            _recheck(root, hashes)


REQUIREMENT_PARAMETERS = {
    'native-observation-source': {'contract', 'model_identity'},
    'native-processor-source': {'expected_source', 'require_same_input_native_reference'},
    'evaluation-cohort': {'table', 'equal_fields', 'document_pattern'},
    'native-run-context': {'equal_fields'},
    'resolved-layout': {'expected_names', 'expected_frame', 'expected_quantity', 'expected_identity', 'atol', 'rtol'},
    'native-named-action': {'expected_action_names', 'expected_driver_namespace', 'require_canonical_matches_driver_return',
                            'require_recorded_matches_driver_return', 'atol', 'rtol'},
}


def _requirements(value):
    """One bounded declarative requirement, delegated to existing family validators."""
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {'kind', 'parameters'} or type(value['kind']) is not str
            or value['kind'] not in REQUIREMENT_PARAMETERS or type(value['parameters']) is not dict
            or set(value['parameters']) - REQUIREMENT_PARAMETERS[value['kind']]):
        raise InputError('Requirements accept supported native kinds and allowlisted parameters only')
    try:
        if len(json.dumps(value, allow_nan=False).encode()) > 64 * 1024:
            raise InputError('Requirements exceed the 64 KiB bound')
        result = deepcopy(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise InputError('Requirements must be bounded finite declarative values') from exc
    parameters = result['parameters']
    if result['kind'] == 'native-observation-source':
        from .native_observation_source import validate_parameters

        result['parameters'] = validate_parameters(parameters)
        return result
    if result['kind'] == 'native-processor-source':
        from .native_processor_source import validate_parameters

        result['parameters'] = validate_parameters(parameters)
        return result
    if result['kind'] == 'evaluation-cohort':
        from .evaluation_cohort import validate_parameters

        result['parameters'] = validate_parameters(parameters)
        return result
    if result['kind'] == 'native-run-context':
        from .native_context import validate_parameters

        result['parameters'] = validate_parameters(parameters)
        return result
    from .layouts import _identity, _tolerances

    _tolerances(parameters.get('atol', 1e-6), parameters.get('rtol', 1e-5))
    if result['kind'] == 'resolved-layout':
        from .layouts import _names, _text

        _names(parameters.get('expected_names'), 'expected_names')
        for key in ('expected_frame', 'expected_quantity'):
            if parameters.get(key) is not None and not _text(parameters[key]):
                raise InputError(key + ' must be bounded explicit text')
        _identity(parameters.get('expected_identity'))
    else:
        from .named_actions import _names, _text

        if parameters.get('expected_action_names') is not None:
            _names(parameters['expected_action_names'])
        if parameters.get('expected_driver_namespace') is not None and not _text(parameters['expected_driver_namespace']):
            raise InputError('Expected driver namespace must be bounded explicit text')
        for key in ('require_canonical_matches_driver_return', 'require_recorded_matches_driver_return'):
            if key in parameters and type(parameters[key]) is not bool:
                raise InputError('Named equality requirements must be strict Boolean values')
    return result


def _requirement_arguments(requirements):
    if requirements is None:
        return {}
    parameters = deepcopy(requirements['parameters'])
    if requirements['kind'] == 'resolved-layout':
        return parameters
    tolerances = {key: parameters.pop(key) for key in ('atol', 'rtol') if key in parameters}
    return {'named_contract': parameters, **tolerances}


def _retained_parameters(requirements, reference):
    parameters, context = deepcopy(requirements['parameters']), {}
    if requirements['kind'] == 'evaluation-cohort':
        return parameters, {'reference': 'Supplied accepted-context native JSON is retained in after/; observed episode keys and selected context only.'}
    if requirements['kind'] == 'native-run-context':
        return parameters, {'reference': 'The supplied reference config is retained in case after/; only selected recorded fields are conserved.'}
    if requirements['kind'] == 'resolved-layout':
        layout = reference['layout']
        inferred = {'expected_frame': layout.get('frame'), 'expected_quantity': layout.get('quantity'),
                    'expected_identity': layout.get('sensor_identity')}
    else:
        inferred = {'expected_action_names': reference.get('action_names'),
                    'expected_driver_namespace': reference.get('driver_namespace')}
        parameters.setdefault('require_canonical_matches_driver_return', False)
        parameters.setdefault('require_recorded_matches_driver_return', False)
    for key, value in inferred.items():
        if parameters.get(key) is None:
            if value is None:
                raise InputError('Known corrected-source context is required for a portable case; none was inferred')
            parameters[key] = deepcopy(value)
            context[key] = deepcopy(value)
    parameters.setdefault('atol', 1e-6)
    parameters.setdefault('rtol', 1e-5)
    return parameters, context


def _creator_recipe(candidate, reference, candidate_path, reference_path, requirements):
    if requirements is None or candidate.get('acceptance_contract', {}).get('status') != 'error':
        return None
    if reference.get('status') != 'pass' or reference.get('acceptance_contract', {}).get('status') != 'pass':
        return None
    if requirements['kind'] == 'resolved-layout':
        before, after = candidate['layout'], reference['layout']
        if (not before.get('sensor_identity') or before['sensor_identity'] != after.get('sensor_identity')
                or set(before['body_names']) != set(after['body_names'])):
            return None
    try:
        parameters, context = _retained_parameters(requirements, reference)
    except InputError:
        return None
    return {'retained_context': context, 'supplied_requirements': deepcopy(requirements),
            'parameter_origins': {key: 'corrected_source_context' if key in context else
                'supplied_requirement' if key in requirements['parameters'] else 'existing_engine_default' for key in parameters},
            'tool': 'create_incident_case', 'arguments': {'check': requirements['kind'].replace('-', '_'),
                'source': candidate_path, 'corrected': reference_path, 'parameters': parameters},
            'required_inputs': ['destination'],
            'scope': 'Choose a new destination and explicitly create this supplied-correction case. Same requirement passed on the reference; no correction was applied and no cause is inferred.'}


def _native_context_investigation(path, reference, requirements):
    from .native_context import audit_native_run_context

    report = audit_native_run_context(path, reference=reference, **requirements['parameters'])
    recipe = None
    actions = []
    if report['status'] == 'error' and reference is not None:
        healthy = audit_native_run_context(reference, reference=reference, **requirements['parameters'])
        configs = {field['config'] for field in report['fields']}
        selection = None
        if len(configs) == 1:
            relative = Path(next(iter(configs))).parent
            candidate_task = Path(path).expanduser().resolve() / relative
            reference_task = Path(reference).expanduser().resolve() / relative
            native_files = report['native_metadata']
            if all(sum(Path(item['path']).parent == folder and
                       bool(re.fullmatch(r'(data|run_[0-9]+)\.hdf5', Path(item['path']).name))
                       for item in native_files[side]) == 1
                   for side, folder in [('candidate', candidate_task), ('reference', reference_task)]):
                selection = candidate_task, reference_task
        intake_unknown = any(finding['status'] == 'unknown' and finding['code'] != 'recorded_context_unknown'
                             for finding in report['findings'])
        if healthy['status'] == 'pass' and selection is not None and not intake_unknown:
            candidate_task, reference_task = selection
            recipe = {'tool': 'create_incident_case', 'arguments': {'check': 'native_run_context',
                'source': str(candidate_task), 'corrected': str(reference_task),
                'parameters': deepcopy(requirements['parameters'])}, 'required_inputs': ['destination'],
                'supplied_requirements': deepcopy(requirements),
                'retained_context': {'reference': 'Original reference config retained in after/; selected recorded fields only.',
                                     'selection': 'Unique matching native task folders selected from the supplied roots.'},
                'scope': 'Retain this comparison criterion and supplied reference. Case creation selects bounded native numeric evidence and explicitly omits media. This is a comparison-selection correction, not a behavioral repair.'}
        elif healthy['status'] == 'pass':
            actions.append({'tool': 'investigate_run_directory', 'arguments': None,
                'purpose': 'Resolve source intake unknowns and select one matching native task folder on each side, containing env_cfg.json and exactly one supported data.hdf5 or run_N.hdf5. A multi-task, config-only or incomplete discovery cannot create this single-task case.',
                'config_candidates': sorted(configs)})
    return {'schema_version': 1, 'status': report['status'], 'root': str(Path(path).expanduser().resolve()),
        'scope': report['scope'], 'inventory': {'native_runs': [str(Path(path).expanduser().resolve())]},
        'coverage': {'executes_code': False, 'model_weights_read': False, 'media_decoded': False,
                     'scope': 'Explicit native-run-context intake uses its own bounded metadata/streamed-hash limits; generic recursive discovery is bypassed.'},
        'audits': [report], 'findings': report.get('findings', []),
        'total_findings': len(report.get('findings', [])), 'findings_truncated': False,
        'next_actions': actions, 'intake_problems': [], 'recording_comparison': None, 'comparison': None,
        'applied_requirements': requirements, 'case_recipe': recipe,
        'summary': 'Checked the supplied recorded-context invariants; outcome labels do not establish policy quality or runtime equivalence.'}


def _evaluation_cohort_investigation(path, reference, requirements):
    from .evaluation_cohort import audit_evaluation_cohort

    report = audit_evaluation_cohort(path, reference=reference, **requirements['parameters'])
    recipe = None
    if report['status'] == 'error' and reference is not None:
        healthy = audit_evaluation_cohort(reference, reference=reference, **requirements['parameters'])
        if healthy['status'] == 'pass':
            recipe = {'tool': 'create_incident_case', 'arguments': {
                'check': 'evaluation_cohort', 'source': str(Path(path).expanduser().resolve()),
                'corrected': str(Path(reference).expanduser().resolve()),
                'parameters': deepcopy(requirements['parameters'])}, 'required_inputs': ['destination'],
                'supplied_requirements': deepcopy(requirements),
                'scope': 'Explicitly nominate the supplied reference as accepted comparison context before creating a case. '
                         'Retains an observed comparison error and the selected criterion, not a policy repair. '
                         'Declared counts, shards, summaries, physical pairing and causal performance remain unverified.'}
    return {'schema_version': 1, 'status': report['status'], 'root': str(Path(path).expanduser().resolve()),
        'scope': report['scope'], 'inventory': {'evaluation_roots': [str(Path(path).expanduser().resolve())]},
        'coverage': {'executes_code': False, 'model_weights_read': False, 'media_decoded': False,
                     'scope': 'Explicit bounded root JSON discovery under the supplied selectors.'},
        'audits': [report], 'findings': report.get('findings', []),
        'total_findings': report['finding_count'], 'findings_truncated': bool(report['findings_omitted']),
        'next_actions': [], 'intake_problems': [], 'recording_comparison': None, 'comparison': None,
        'applied_requirements': requirements, 'case_recipe': recipe,
        'summary': 'Compared observed episode identities, Boolean outcomes and selected native context. '
                   'This does not validate declared completeness or establish causal model superiority.'}


def _is_native_observation_source(path):
    """Recognize only the fixed root declaration; never search its internals."""
    root = Path(path).expanduser()
    if root.is_symlink():
        return False
    declaration = root / 'capture.json'
    if declaration.is_symlink() or not declaration.is_file():
        return False
    try:
        value = read_json(declaration)
    except InputError:
        return False
    return type(value) is dict and value.get('kind') == 'native_observation_source_capture'


def _native_observation_source_investigation(path, reference, requirements):
    from .native_observation_source import (
        audit_native_observation_source,
        bind_native_observation_source_parameters,
    )

    original = Path(path).expanduser()
    if original.is_symlink():
        raise InputError('Native source capture root symlink refused')
    root = str(original.resolve())
    parameters = deepcopy(requirements['parameters'])
    healthy = None
    if reference is not None:
        if Path(reference).expanduser().is_symlink():
            raise InputError('Native reference capture root symlink refused')
        parameters = bind_native_observation_source_parameters(reference, parameters or None)
    report = audit_native_observation_source(root, **parameters)
    if not parameters and report['status'] in ('error', 'pass'):
        parameters = bind_native_observation_source_parameters(root)
    if reference is not None:
        healthy = audit_native_observation_source(reference, **parameters)
    recipe = None
    contradiction = report['source_criterion'].get('code') == 'source_contradiction'
    same_correction_input = (type(report.get('input_digest')) is str
        and len(report['input_digest']) == 64 and healthy is not None
        and report['input_digest'] == healthy.get('input_digest')
        and type(report.get('native_identity')) is dict and bool(report['native_identity'])
        and report['native_identity'] == healthy.get('native_identity')
        and report.get('model_identity') is not None and report['model_identity'] == healthy.get('model_identity'))
    if (report['status'] == 'error' and contradiction and healthy is not None
            and healthy['status'] == 'pass' and same_correction_input):
        recipe = {'tool': 'create_incident_case', 'arguments': {
            'check': 'native_observation_source', 'source': root,
            'corrected': str(Path(reference).expanduser().resolve()),
            'parameters': deepcopy(parameters)}, 'required_inputs': ['destination'],
            'supplied_requirements': deepcopy(requirements),
            'scope': 'Retains the declared input-source contradiction and same-input supplied native correction. '
                     'Case reproduction checks recorded evidence; it does not run native preprocessing or a policy.'}
    comparison = None if healthy is None else {
        'kind': 'native_observation_source_comparison', 'status': report['status'],
        'comparison_supported': True, 'difference_is_fault': contradiction,
        'first_supported_difference': report['source_criterion'] if contradiction else None,
        'reference_audit_status': healthy['status'],
        'same_correction_input': same_correction_input,
        'scope': 'Same declared input contract for both captures; each binds its own original-derived current input.'}
    return {'schema_version': 1, 'status': report['status'], 'root': root,
        'scope': report['scope'], 'inventory': {'captures': [root], 'checkpoints': [],
            'datasets': [], 'recordings': [], 'trajectories': [], 'dispatch_evidence': []},
        'coverage': {'files_seen': 1, 'directories_seen': 1, 'bytes_inspected': 0,
            'symlinks_skipped': 0, 'truncated': False, 'executes_code': False,
            'model_weights_read': False, 'media_decoded': False,
            'scope': 'Fixed recorded source capture delegated to its bounded evidence verifier.'},
        'audits': [report], 'findings': [dict(item, status='error', code=item['kind'],
            message='Declared native input value was not preserved: ' + str(item.get('group', item.get('camera', 'unknown'))))
            for item in report.get('findings', [])],
        'total_findings': report['finding_count'], 'findings_truncated': bool(report['findings_omitted']),
        'next_actions': [], 'intake_problems': [], 'recording_comparison': None,
        'comparison': comparison, 'applied_requirements': {
            'kind': 'native-observation-source', 'parameters': parameters}, 'case_recipe': recipe,
        'summary': 'Checked declared pre-normalization state/camera API values against current original-derived inputs. '
                   'Native numerical replay and policy competence are separate.'}



def _is_native_processor_source(path):
    """Recognize only the fixed root declaration; never search its internals."""
    root = Path(path).expanduser()
    if root.is_symlink():
        return False
    declaration = root / 'capture.json'
    if declaration.is_symlink() or not declaration.is_file():
        return False
    try:
        value = read_json(declaration)
    except InputError:
        return False
    return type(value) is dict and value.get('kind') == 'native_processor_source_capture'


def _native_processor_source_investigation(path, reference, requirements):
    from .native_processor_source import (
        audit_native_processor_source,
        bind_native_processor_source_parameters,
    )

    original = Path(path).expanduser()
    if original.is_symlink():
        raise InputError('Native source capture root symlink refused')
    root = str(original.resolve())
    parameters = deepcopy(requirements['parameters'])
    healthy = None
    if reference is not None:
        if Path(reference).expanduser().is_symlink():
            raise InputError('Native reference capture root symlink refused')
        parameters = bind_native_processor_source_parameters(reference, parameters)
    report = audit_native_processor_source(root, **parameters)
    if reference is not None:
        healthy = audit_native_processor_source(reference, **parameters)
    recipe = None
    contradiction = report['source_criterion'].get('code') == 'source_contradiction'
    same_correction_input = (type(report.get('input_digest')) is str
        and len(report['input_digest']) == 64 and healthy is not None
        and report['input_digest'] == healthy.get('input_digest')
        and type(report.get('native_identity')) is dict and bool(report['native_identity'])
        and report['native_identity'] == healthy.get('native_identity'))
    if (report['status'] == 'error' and contradiction and healthy is not None
            and healthy['status'] == 'pass' and same_correction_input):
        recipe = {'tool': 'create_incident_case', 'arguments': {
            'check': 'native_processor_source', 'source': root,
            'corrected': str(Path(reference).expanduser().resolve()),
            'parameters': deepcopy(parameters)}, 'required_inputs': ['destination'],
            'supplied_requirements': deepcopy(requirements),
            'scope': 'Retains the observed caller source contradiction and same-input selected-source correction. '
                     'Case reproduction checks recorded evidence; it does not run native preprocessing or a policy.'}
    comparison = None if healthy is None else {
        'kind': 'native_processor_source_comparison', 'status': report['status'],
        'comparison_supported': True, 'difference_is_fault': contradiction,
        'first_supported_difference': report['source_criterion'] if contradiction else None,
        'reference_audit_status': healthy['status'],
        'same_correction_input': same_correction_input,
        'scope': 'Same source criterion for both captures; each native reference binds its own current input.'}
    return {'schema_version': 1, 'status': report['status'], 'root': root,
        'scope': report['scope'], 'inventory': {'captures': [root], 'checkpoints': [],
            'datasets': [], 'recordings': [], 'trajectories': [], 'dispatch_evidence': []},
        'coverage': {'files_seen': 1, 'directories_seen': 1, 'bytes_inspected': 0,
            'symlinks_skipped': 0, 'truncated': False, 'executes_code': False,
            'model_weights_read': False, 'media_decoded': False,
            'scope': 'Fixed recorded source capture delegated to its bounded evidence verifier.'},
        'audits': [report], 'findings': report.get('findings', []),
        'total_findings': report['finding_count'], 'findings_truncated': bool(report['findings_omitted']),
        'next_actions': [], 'intake_problems': [], 'recording_comparison': None,
        'comparison': comparison, 'applied_requirements': {
            'kind': 'native-processor-source', 'parameters': parameters}, 'case_recipe': recipe,
        'summary': 'Checked recorded processor-source association and current-input reference evidence. '
                   'Native numerical replay and policy competence are separate.'}


RECOGNIZED_JSON = {'config.json', 'capture.json', 'selection.json', 'manifest.json', 'READY.json'}
TABULAR = ('.npz', '.csv', '.parquet', '.h5', '.hdf5')
# A generic NPZ/CSV counts as a rollout recording only with an action stream ('action', 'actions', 'action.x',
# 'joint_action', ...): the native LeRobot Parquet and robomimic HDF5 readers already require one.
_ACTION_STREAM = re.compile(r'(^|[._/\[ -])actions?($|[._/\[ -])', re.IGNORECASE)
# inspect_rollout refusals that still identify a native rollout which needs an explicit episode selection.
_SELECTION_NEEDED = ('Multiple native episodes',)
_EXTRA_NEEDED = ('Install policy-deployment-doctor[',)
NO_SUPPORTED_ARTIFACT = ('No supported robot artifact was found here. RoboClue reads LeRobot datasets (a folder with '
                         'meta/info.json) and checkpoints (config.json), runtime captures, rollout recordings with an '
                         'action stream (LeRobot Parquet, robomimic HDF5, NPZ or CSV), ROS trajectories and retained '
                         'cases.')
NO_ARTIFACT_INCOMPLETE = ('No supported robot artifact was recognized here, but discovery was incomplete (files over '
                          'the read bounds, skipped links, intake problems or a truncated walk); see coverage and '
                          'intake_problems before concluding there is none.')
LARGE_NOT_READ = 'Bounded discovery lists large recordings without decoding them.'


def _header_streams(file: Path, fmt: str) -> list[str] | None:
    """Complete bounded stream names of a large NPZ/CSV, or None when introspection is incomplete.

    A prefix without an action name cannot establish that the whole header/archive has no action stream.
    """
    import csv
    import io
    import zipfile

    try:
        fd = os.open(file, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
        with os.fdopen(fd, 'rb') as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            if fmt == '.csv':
                raw = handle.readline(64 * 1024 + 1)
                if len(raw) > 64 * 1024:
                    return None
                # Strict parsing refuses a quoted header record that continues beyond this physical line.
                names = next(csv.reader(io.StringIO(raw.decode('utf-8-sig')), strict=True), [])
                return names if len(names) <= 4096 else None
            with zipfile.ZipFile(handle) as archive:
                names = archive.namelist()[:4097]
                if len(names) > 4096:
                    return None
                return [name[:-4] if name.endswith('.npy') else name for name in names]
    except (OSError, ValueError, UnicodeError, EOFError, RuntimeError, NotImplementedError, csv.Error,
            zipfile.BadZipFile):
        return None


def _shown(names: list[str]) -> str:
    return ', '.join(n.strip()[:64] for n in names[:8]) + (', ...' if len(names) > 8 else '')


def _tabular_role(item: dict, header: list[str] | None = None) -> tuple[str, str | None]:
    """'recording', 'unrecognized' or 'not_inspected', with the reason for the last two."""
    if 'inventory' in item:
        names = [str(name) for name in item['inventory'].get('streams', {})]
        if item['format'] in ('.parquet', '.h5', '.hdf5') or any(_ACTION_STREAM.search(n.strip()) for n in names):
            return 'recording', None
        return 'unrecognized', f"numeric {item['format'][1:].upper()} without an action stream (streams: {_shown(names)})"
    reason = item.get('needs_selection_or_adapter')
    if reason is not None:
        if reason.startswith(_SELECTION_NEEDED):
            return 'recording', None
        if reason.startswith(_EXTRA_NEEDED):
            return 'not_inspected', reason
        return 'unrecognized', reason
    # Over the read bound: native LeRobot Parquet / robomimic HDF5 stay listed by format, as before; a large NPZ/CSV is
    # judged from its stream names only (zip directory or header line), never decoded.
    if item['format'] in ('.parquet', '.h5', '.hdf5'):
        return 'recording', None
    if header is not None:
        if any(_ACTION_STREAM.search(n.strip()) for n in header):
            return 'recording', None
        return 'unrecognized', (f"{item['format'][1:].upper()} without an action stream (streams from the "
                                f"{'header line' if item['format'] == '.csv' else 'archive directory'}, not decoded: "
                                f"{_shown(header)})")
    return 'not_inspected', ('Beyond the 1 MiB per-file (16 MiB total) discovery read bound; its stream names could '
                             'not be completely read (64 KiB CSV header / 4096 names, or unreadable names). '
                             'Pass it to compare_rollout_recordings directly if it is a rollout recording.')


def _recognized_name(file: Path) -> bool:
    """File names discovery reads as candidates; a non-object JSON under one of them is corrupt, not unrelated."""
    name = file.name.lower()
    return (file.name in RECOGNIZED_JSON or (file.name == 'info.json' and file.parent.name == 'meta')
            or 'deployment' in name or 'contract' in name)


def investigate_run(path: str, *, reference: str | None = None, named_contract=None, requirements=None, max_files: int = 256, max_depth: int = 4) -> dict:
    if type(max_files) is not int or not 1 <= max_files <= 1000 or type(max_depth) is not int or not 0 <= max_depth <= 8:
        raise InputError('Require max_files in [1,1000], max_depth in [0,8]')
    if named_contract is not None and requirements is not None:
        raise InputError('Choose requirements or named_contract; do not provide them together')
    requirements = _requirements(requirements)
    if requirements is not None and requirements['kind'] == 'native-observation-source':
        return _native_observation_source_investigation(path, reference, requirements)
    if requirements is None and _is_native_observation_source(path):
        return _native_observation_source_investigation(path, reference,
            {'kind': 'native-observation-source', 'parameters': {}})
    if requirements is not None and requirements['kind'] == 'native-processor-source':
        return _native_processor_source_investigation(path, reference, requirements)
    if requirements is None and _is_native_processor_source(path):
        if named_contract is not None:
            raise InputError('Named-action criteria do not apply to a processor-source capture')
        return _native_processor_source_investigation(path, reference,
            _requirements({'kind': 'native-processor-source', 'parameters': {}}))
    if requirements is not None and requirements['kind'] == 'evaluation-cohort':
        return _evaluation_cohort_investigation(path, reference, requirements)
    if requirements is not None and requirements['kind'] == 'native-run-context':
        return _native_context_investigation(path, reference, requirements)
    if named_contract is not None:
        allowed = {'expected_action_names', 'expected_driver_namespace', 'require_canonical_matches_driver_return', 'require_recorded_matches_driver_return'}
        if type(named_contract) is not dict or set(named_contract) - allowed:
            raise InputError('Investigate named contract accepts only existing named-action criteria')
        from .named_actions import _names, _text

        if named_contract.get('expected_action_names') is not None:
            _names(named_contract['expected_action_names'])
        if named_contract.get('expected_driver_namespace') is not None and not _text(named_contract['expected_driver_namespace']):
            raise InputError('Expected driver namespace must be bounded explicit text')
        for key in ('require_canonical_matches_driver_return', 'require_recorded_matches_driver_return'):
            if key in named_contract and type(named_contract[key]) is not bool:
                raise InputError('Named equality requirements must be strict Boolean values')
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise InputError(f'investigate requires an existing run directory: {path!r}')
    inventory = {'checkpoints': [], 'datasets': [], 'deployments': [], 'captures': [], 'recordings': [],
                 'saved_outputs': [], 'trajectories': [], 'dispatch_evidence': [], 'incident_cases': [],
                 'retained_checks': [], 'trajectory_cases': [], 'incomplete_cases': []}
    coverage = {'files_seen': 0, 'directories_seen': 0, 'bytes_inspected': 0, 'non_object_json_skipped': [],
                'symlinks_skipped': 0, 'truncated': False, 'executes_code': False,
                'model_weights_read': False, 'max_files': max_files, 'max_depth': max_depth}
    problems, queue, entries_seen, manifest_cache = [], [(root, 0)], 0, {}
    tabular = {'unrecognized': [], 'not_inspected': []}
    while queue:
        directory, depth = queue.pop(0)
        atomic = inventory['captures'] + inventory['datasets'] + inventory['incident_cases']
        if any(directory != Path(candidate) and directory.is_relative_to(Path(candidate)) for candidate in atomic):
            continue
        coverage['directories_seen'] += 1
        if coverage['directories_seen'] > 128:
            coverage['truncated'] = True
            break
        if _discover_case_boundary(directory, inventory, coverage, problems, max_files, manifest_cache):
            continue
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries_seen += 1
                    cached_manifest = manifest_cache.get(entry.path)
                    if entries_seen > max_files + 256 or (coverage['files_seen'] >= max_files and cached_manifest is None):
                        coverage['truncated'] = True
                        break
                    if entry.is_symlink():
                        coverage['symlinks_skipped'] += 1
                        continue
                    file = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in SKIP and not entry.name.startswith('.'):
                            if depth < max_depth:
                                queue.append((file, depth + 1))
                            else:
                                coverage['truncated'] = True
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if cached_manifest is None:
                        coverage['files_seen'] += 1
                    size = entry.stat(follow_symlinks=False).st_size
                    if file.suffix.lower() in TABULAR:
                        item = {'path': str(file), 'format': file.suffix.lower(), 'size_bytes': size}
                        if size <= 1024 * 1024 and coverage['bytes_inspected'] + size <= 16 * 1024 * 1024:
                            coverage['bytes_inspected'] += size
                            try:
                                item['inventory'] = inspect_rollout(str(file))
                            except InputError as exc:
                                item['needs_selection_or_adapter'] = str(exc)
                        else:
                            item['inventory_not_read'] = LARGE_NOT_READ
                        header = (_header_streams(file, item['format'])
                                  if 'inventory_not_read' in item and item['format'] in ('.npz', '.csv') else None)
                        role, reason = _tabular_role(item, header)
                        if role == 'recording':
                            inventory['recordings'].append(item)
                        else:  # listed in coverage, never counted or suggested as a recording
                            tabular[role].append({'path': str(file), 'format': item['format'], 'size_bytes': size,
                                                  'reason': reason})
                    if file.suffix.lower() not in ('.json', '.yaml', '.yml'):
                        continue
                    if size > 1024 * 1024 or coverage['bytes_inspected'] + size > 16 * 1024 * 1024:
                        coverage['truncated'] = True
                        continue
                    if cached_manifest is None:
                        coverage['bytes_inspected'] += size
                    try:
                        data = (cached_manifest[0] if cached_manifest is not None else
                                read_trajectory_document(file)[0] if file.suffix.lower() in ('.yaml', '.yml') else read_json_value(file))
                    except InputError as exc:
                        problems.append({'path': str(file), 'reason': str(exc)})
                        continue
                    if not isinstance(data, dict):
                        # Decided from the value already parsed in the single bounded read above. A JSON list (e.g. a
                        # Hub file listing) under a name discovery does not recognize cannot be a candidate: listed and
                        # skipped. A null or scalar, or any non-object under a recognized name, may be a corrupt
                        # candidate and stays an intake problem. YAML documents keep their earlier handling.
                        if file.suffix.lower() == '.json' and cached_manifest is None:
                            if isinstance(data, list) and not _recognized_name(file):
                                if len(coverage['non_object_json_skipped']) < 20:
                                    coverage['non_object_json_skipped'].append(str(file))
                            else:
                                problems.append({'path': str(file), 'reason': f'Expected a JSON object: {file}'})
                        continue
                    if ('joint_names' in data and 'points' in data) or any(
                            isinstance(data.get(key), dict) and 'joint_names' in data[key] and 'points' in data[key]
                            for key in ('joint_trajectory', 'trajectory')):
                        if coverage['bytes_inspected'] + size > 16 * 1024 * 1024:
                            coverage['truncated'] = True
                            continue
                        coverage['bytes_inspected'] += size
                        try:
                            inventory['trajectories'].append(_trajectory_candidate(str(file)))
                        except InputError as exc:
                            inventory['trajectories'].append({'path': str(file), 'needs_selection_or_adapter': str(exc)})
                            problems.append({'path': str(file), 'reason': str(exc)})
                    elif data.get('kind') == 'action_dispatch_evidence':
                        inventory['dispatch_evidence'].append(str(file))
                    elif file.name == 'manifest.json' and data.get('kind') == 'robot_incident_case':
                        inventory['incident_cases'].append(str(file.parent))
                    elif file.name == 'selection.json' and data.get('kind') == 'act_selection_capture':
                        inventory['captures'].append(str(file.parent))
                    elif file.name == 'capture.json':
                        inventory['captures'].append(str(file.parent))
                    elif file.name == 'config.json' and isinstance(data.get('type'), str) and ('input_features' in data or 'output_features' in data):
                        inventory['checkpoints'].append(str(file.parent))
                    elif file.name == 'info.json' and file.parent.name == 'meta' and isinstance(data.get('features'), dict):
                        dataset = file.parent.parent
                        if dataset.is_relative_to(root):
                            inventory['datasets'].append(str(dataset))
                        else:
                            problems.append({'path': str(file), 'reason': 'Dataset directory expands outside discovery root; not read.'})
                    elif type(data.get('schema_version')) is int and data['schema_version'] == 1 and isinstance(data.get('features'), dict):
                        inventory['deployments'].append(str(file))
                    elif isinstance(data.get('model_output'), list):
                        inventory['saved_outputs'].append(str(file))
        except OSError as exc:
            problems.append({'path': str(directory), 'reason': str(exc)})
        if coverage['truncated'] and coverage['files_seen'] >= max_files:
            break
    for key in ('checkpoints', 'datasets', 'deployments', 'captures', 'saved_outputs'):
        inventory[key] = sorted(set(inventory[key]))
    # Files inside a discovered dataset (meta/tasks.parquet, data shards) belong to that dataset, not to rollouts.
    inventory['recordings'] = sorted((item for item in inventory['recordings']
                                      if not any(Path(item['path']).is_relative_to(Path(d)) for d in inventory['datasets'])),
                                     key=lambda item: item['path'])
    for role, items in tabular.items():
        items = sorted((item for item in items
                        if not any(Path(item['path']).is_relative_to(Path(d)) for d in inventory['datasets'])),
                       key=lambda item: item['path'])
        coverage[f'tabular_files_{role}'] = items[:20]
        coverage[f'tabular_files_{role}_count'] = len(items)
    inventory['trajectories'].sort(key=lambda item: item['path'])
    inventory['dispatch_evidence'].sort()
    captures = inventory['captures']
    # A capture's checkpoint is not a second independent candidate.
    ordinary = [cp for cp in inventory['checkpoints'] if not any(Path(cp).is_relative_to(Path(capture)) for capture in captures)]
    audits, actions, findings = [], [], []
    dataset = inventory['datasets'][0] if len(inventory['datasets']) == 1 else None
    ambiguous = len(captures) + len(ordinary) > 1 or len(inventory['datasets']) > 1 or len(inventory['deployments']) > 1
    incomplete = coverage['truncated'] or bool(problems)
    if not ambiguous and not incomplete and (captures or ordinary):
        if requirements is not None:
            if not captures:
                raise InputError('Requirements need a matching native capture family')
            manifest_kind = read_json(Path(captures[0]) / 'capture.json').get('kind')
            expected_kind = {'resolved-layout': 'resolved_layout_capture', 'native-named-action': 'named_action_capture'}[requirements['kind']]
            if manifest_kind != expected_kind:
                raise InputError('Requirements kind does not match the discovered native capture family')
        try:
            if captures:
                args = {'capture': captures[0]}
                manifest_file = Path(captures[0]) / 'capture.json'
                manifest = read_json(manifest_file) if manifest_file.exists() else {'kind': 'act_selection_capture'}
                if manifest.get('kind') not in ('named_action_capture', 'command_path_capture', 'resolved_layout_capture', 'act_selection_capture') and dataset:
                    args['dataset'] = dataset
                if named_contract is not None:
                    args['named_contract'] = named_contract
                args.update(_requirement_arguments(requirements))
                report = audit_capture(**args)
                actions.append({'tool': 'audit_runtime_capture', 'arguments': args,
                                'purpose': 'Verify captured effective bytes and object-linkage observations.'})
            else:
                args = {'checkpoint': ordinary[0]}
                if named_contract is not None:
                    raise InputError('Named criteria require a native named-action capture, not a checkpoint')
                if dataset:
                    args['dataset'] = dataset
                if inventory['deployments']:
                    args['deployment'] = inventory['deployments'][0]
                report = audit(**args)
                actions.append({'tool': 'audit_deployment', 'arguments': args,
                                'purpose': 'Reproduce this contract audit on the discovered unique candidates.'})
            findings = _capture_findings(report)
            audits.append(_capture_summary(report))
        except (InputError, OSError) as exc:
            problems.append({'path': str(root), 'reason': str(exc)})
            actions.append({'tool': 'audit_runtime_capture' if captures else 'audit_deployment',
                            'arguments': args,
                            'purpose': 'Audit failed to read or verify evidence. Review the intake problem; for a changed capture, export a fresh snapshot from the trusted running process, then retry.'})
    for case in inventory['incident_cases'][:8]:
        actions.append({'tool': 'reproduce_incident_case', 'arguments': {'directory': case},
                        'purpose': 'Verify this existing portable incident case and its retained controls/verifier; discovery has not reproduced it.'})
    for case in inventory['retained_checks'][:8]:
        actions.append({'tool': 'reproduce_incident_case', 'arguments': {'directory': case},
                        'purpose': 'Verify and reproduce this retained metadata/recording/video-reference check with its full findings and verifier; discovery has not verified or reproduced it.'})
    for case in inventory['trajectory_cases'][:8]:
        actions.append({'tool': 'reproduce_trajectory_case', 'arguments': {'directory': case},
                        'purpose': 'Verify this retained trajectory check; discovery has not verified its evidence or findings.'})
    for case in inventory['incomplete_cases'][:8]:
        actions.append({'tool': 'reproduce_incident_case', 'arguments': {'directory': case},
                        'purpose': 'This incomplete case has READY but no manifest. Explicit replay must refuse it until the complete original case is supplied; discovery has not inspected its internal inputs.'})
    comparison, case_recipe = None, None
    if reference is not None:
        reference_report = investigate_run(reference, named_contract=named_contract, requirements=requirements, max_files=max_files, max_depth=max_depth)
        reference_captures = reference_report['inventory']['captures']
        tolerances = {key: requirements['parameters'][key] for key in ('atol', 'rtol') if requirements is not None and key in requirements['parameters']}
        if (len(captures) == len(reference_captures) == 1 and not ambiguous and not incomplete
                and len(reference_report['audits']) == 1
                and not reference_report['coverage']['truncated'] and not reference_report['intake_problems']):
            kinds = [read_json(Path(value) / ('capture.json' if (Path(value) / 'capture.json').exists() else 'selection.json')).get('kind')
                     for value in (reference_captures[0], captures[0])]
            if kinds == ['named_action_capture', 'named_action_capture']:
                try:
                    comparison = _boundary_comparison(reference_captures[0], captures[0], **tolerances)
                except (InputError, OSError) as exc:
                    problems.append({'path': str(root), 'reason': str(exc)})
            elif kinds[0] == kinds[1]:
                from .runtime_diff import compare_runtime_captures

                try:
                    comparison = compare_runtime_captures(reference_captures[0], captures[0], **tolerances)
                except (InputError, OSError) as exc:
                    problems.append({'path': str(root), 'reason': str(exc)})
        if comparison is not None and len(reference_report['audits']) == 1:
            comparison['reference_audit_status'] = reference_report['audits'][0]['status']
            comparison['reference_criterion_status'] = reference_report['audits'][0].get('acceptance_contract', {}).get('status', 'unknown')
            if len(audits) == 1:
                case_recipe = _creator_recipe(audits[0], reference_report['audits'][0], captures[0], reference_captures[0], requirements)
        if comparison is None:
            comparison = {'kind': 'run_boundary_comparison', 'status': 'unknown', 'comparison_supported': False,
                          'difference_is_fault': False, 'first_supported_difference': None,
                          'reason': 'Choose exactly one complete named-action capture per explicit run; other families retain their existing comparison tools.'}
    if ambiguous or incomplete:
        actions.append({'tool': 'audit_deployment', 'arguments': None,
                        'purpose': 'Choose the intended checkpoint/capture, dataset and runtime contract; incomplete discovery or multiple candidates cannot establish unique association.'})
    if len(inventory['recordings']) >= 2:
        actions.append({'tool': 'compare_rollout_recordings', 'arguments': None,
                        'candidates': [r['path'] for r in inventory['recordings'][:8]],
                        'purpose': 'Choose reference/candidate and native episode IDs. Supply an alignment index; no pairing or clock alignment is invented.'})
    if inventory['saved_outputs']:
        actions.append({'tool': 'trace_saved_actions', 'arguments': None,
                        'candidates': inventory['saved_outputs'][:8],
                        'purpose': 'Choose the effective checkpoint/capture and saved model-output sample to test decoding.'})
    for candidate in inventory['datasets'][:8]:
        actions.append({'tool': 'preflight_check', 'arguments': {'source': candidate},
                        'purpose': 'Run the combined dataset checks this discovery does not run: rows, episode spans, '
                                   'video index and saved statistics (CLI: policy-doctor preflight DATASET).'})
        actions.append({'tool': 'audit_dataset_windows', 'arguments': {'dataset': candidate},
                        'purpose': 'For supported native v3 datasets, check episode metadata against actual row membership before trusting temporal action windows. Numeric row equality alone does not qualify this association.'})
    for candidate in inventory['trajectories'][:8]:
        if candidate.get('suggested_sample_times'):
            actions.append({'tool': 'sample_ros_trajectory',
                            'arguments': {'trajectory': candidate['path'],
                                          'sample_times': candidate['suggested_sample_times'],
                                          'interpolation_method': 'splines'},
                            'purpose': candidate['suggestion_scope'] + ' ' + candidate['interpolation_scope']})
    if len(inventory['trajectories']) >= 2:
        actions.append({'tool': 'compare_ros_trajectories', 'arguments': None,
                        'candidates': [c['path'] for c in inventory['trajectories'][:8]],
                        'purpose': 'Choose the actual reference/candidate, sample times and each interpolation method; filenames and colocation do not establish a pair.'})
    for evidence in inventory['dispatch_evidence'][:8]:
        actions.append({'tool': 'audit_action_boundary', 'arguments': {'evidence': evidence},
                        'purpose': 'Inspect declared boundary evidence without a recording association. Supply the actual finalized shard, episode and action names to check a recording; discovery does not validate this sidecar.'})
    unverified_case_count = sum(len(inventory[key]) for key in
                                ('incident_cases', 'retained_checks', 'trajectory_cases', 'incomplete_cases'))
    nothing_found = not audits and not any(inventory.values())
    incomplete_discovery = bool(problems or coverage['truncated'] or coverage['symlinks_skipped']
                                or coverage['tabular_files_not_inspected_count'])
    nothing_supported = nothing_found and not incomplete_discovery
    lead = ((NO_SUPPORTED_ARTIFACT if nothing_supported else NO_ARTIFACT_INCOMPLETE) + ' ') if nothing_found else ''
    other_tabular = ''
    if coverage['tabular_files_unrecognized_count']:
        other_tabular += (f" {coverage['tabular_files_unrecognized_count']} tabular file(s) are not rollout recordings "
                          "RoboClue recognizes (no action stream or an unsupported layout); they are listed in coverage.")
    if coverage['tabular_files_not_inspected_count']:
        other_tabular += (f" {coverage['tabular_files_not_inspected_count']} tabular file(s) were not inspected "
                          "(size bound or missing extra); they are listed in coverage with the reason.")
    status = 'error' if any(a['status'] == 'error' for a in audits) else 'unknown' if (
        not audits or ambiguous or problems or coverage['truncated'] or unverified_case_count or any(a['status'] == 'unknown' for a in audits)) else 'pass'
    return {'schema_version': 1, 'status': status, 'root': str(root),
            'scope': 'Bounded offline discovery and contract triage. Capture/dataset/case internals are delegated to their existing engines, not recursively discovered. Unique colocated candidates are a selection assumption, not proof of runtime association or causation.',
            'inventory': inventory, 'coverage': coverage, 'audits': audits, 'unverified_case_count': unverified_case_count,
            'findings': findings[:64], 'total_findings': len(findings), 'findings_truncated': len(findings) > 64,
            'next_actions': actions, 'intake_problems': problems[:20], 'recording_comparison': None, 'comparison': comparison, 'applied_requirements': requirements, 'case_recipe': case_recipe,
            'no_supported_artifact': nothing_supported,
            'summary': lead + f"Found {len(inventory['checkpoints'])} checkpoint(s), {len(captures)} capture(s), {len(inventory['datasets'])} dataset(s), {len(inventory['recordings'])} recording(s), {len(inventory['trajectories'])} trajectory message(s), {len(inventory['dispatch_evidence'])} boundary trace(s), {len(inventory['retained_checks'])} retained check(s); ran {len(audits)} unambiguous audit(s)." +
                       (f" {unverified_case_count} portable case(s) remain unverified until explicit replay." if unverified_case_count else '') +
                       (" Dataset checks did not run here; run policy-doctor preflight on each dataset." if inventory['datasets'] else '')
                       + other_tabular}
