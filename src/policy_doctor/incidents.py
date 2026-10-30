"""One native incident lifecycle: retained fault, reviewed correction and controls.

Only allowlisted local diagnostic engines run. Native inputs are copied, never edited.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath

from . import __version__
from .artifacts import InputError, finite_tree, read_json
from .cases import _encode, _portable
from .publication import publish_capture
from .trajectory import sample_joint_trajectory
from .windows import audit_episode_windows

KIND = 'robot_incident_case'
CHECKS = {'episode_windows', 'trajectory_sampling', 'resolved_layout', 'act_query_cadence', 'act_command_path', 'native_named_action', 'native_run_context', 'evaluation_cohort', 'native_processor_source', 'native_observation_source'}
PARAMETERS = {'native_observation_source': {'contract', 'model_identity'},
              'native_processor_source': {'expected_source', 'require_same_input_native_reference'},
              'evaluation_cohort': {'table', 'equal_fields', 'document_pattern'},
              'native_run_context': {'equal_fields'},
              'native_named_action': {'expected_action_names', 'expected_driver_namespace', 'require_canonical_matches_driver_return', 'require_recorded_matches_driver_return', 'atol', 'rtol'},
              'act_command_path': {'expected_processor', 'expected_action_names', 'expected_consumer_namespace', 'atol', 'rtol'},
              'act_query_cadence': {'max_query_gap', 'atol', 'rtol'},
              'episode_windows': {'episode_index', 'anchor_index', 'delta_indices', 'max_rows', 'max_files'},
              'trajectory_sampling': {'sample_times', 'interpolation_method', 'position_limits'},
              'resolved_layout': {'expected_names', 'expected_frame', 'expected_quantity', 'expected_identity', 'atol', 'rtol'}}
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_PAYLOAD_FILES = 256
SCOPE = ('Reproduces a retained native diagnostic error, a corrected-artifact pass and named controls. '
         'A reproduced case is not a healthy robot, physical repair or author-authenticity certificate. '
         'Only allowlisted offline checks run; native source inputs are never modified.')


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(path):
    size = 0
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            while chunk := stream.read(512 * 1024):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise InputError('Incident input exceeds the 128 MiB per-file budget')
                digest.update(chunk)
    except OSError as exc:
        raise InputError(f'Cannot read incident evidence: {exc}') from exc
    return digest.hexdigest(), size


def _inside(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative or
            str(PurePosixPath(relative)) != relative or PurePosixPath(relative).is_absolute() or
            any(p in ('.', '..') for p in PurePosixPath(relative).parts)):
        raise InputError('Incident file locators must be normalized relative paths')
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink() or not current.resolve().is_relative_to(root):
            raise InputError('Incident evidence symlink/root escape refused')
    return current


def _inventory(root, *, case=False):
    """Bounded physical tree inventory; do not follow a symlink even to enumerate it."""
    result, queue, entries, total = {}, [(root, 0)], 0, 0
    while queue:
        directory, depth = queue.pop()
        if depth > 8:
            raise InputError('Incident evidence exceeds the directory-depth budget')
        try:
            with os.scandir(directory) as stream:
                for item in stream:
                    entries += 1
                    if entries > 2048:
                        raise InputError('Incident evidence exceeds the directory-entry budget')
                    path = _inside(root, str(Path(item.path).relative_to(root)))
                    if item.is_dir(follow_symlinks=False):
                        queue.append((path, depth + 1))
                    elif item.is_file(follow_symlinks=False):
                        if path.stat().st_size > MAX_FILE_BYTES:
                            raise InputError('Incident input exceeds the per-file byte budget')
                        digest, size = _hash(path)
                        total += size
                        if total > MAX_TOTAL_BYTES or len(result) >= MAX_PAYLOAD_FILES + (3 if case else 0):
                            raise InputError('Incident payload file/byte budget exceeded')
                        result[str(path.relative_to(root))] = (digest, size)
                    else:
                        raise InputError('Incident evidence must contain only regular files/directories')
        except OSError as exc:
            raise InputError(f'Cannot enumerate incident evidence: {exc}') from exc
    return result


def _native_source(path, check):
    if not isinstance(path, (str, Path)) or not str(path):
        raise InputError('Require a native source path')
    source = Path(path).expanduser()
    if check == 'episode_windows' and source.name == 'info.json' and source.parent.name == 'meta':
        if source.is_symlink() or source.parent.is_symlink():
            raise InputError('Native info.json alias symlink refused')
        source = source.parent.parent
    if source.is_symlink():
        raise InputError('Incident source symlink refused')
    source = source.resolve()
    if check in ('episode_windows', 'resolved_layout', 'act_query_cadence', 'act_command_path', 'native_named_action', 'native_run_context', 'evaluation_cohort', 'native_processor_source', 'native_observation_source') and not source.is_dir():
        raise InputError(f'{check} needs a native evidence directory at {str(path)!r}; check the source path and rerun')
    if check == 'trajectory_sampling' and (not source.is_file() or source.suffix.lower() not in ('.json', '.yaml', '.yml')):
        raise InputError('trajectory_sampling needs one native JSON/YAML message')
    return source


def _parameters(check, parameters):
    if not isinstance(parameters, dict) or set(parameters) - PARAMETERS[check]:
        raise InputError('Only allowlisted incident check parameters are supported')
    if check == 'native_observation_source':
        from .native_observation_source import validate_parameters

        return validate_parameters(parameters)
    if check == 'native_processor_source':
        from .native_processor_source import validate_parameters

        return validate_parameters(parameters)
    if check == 'evaluation_cohort':
        from .evaluation_cohort import validate_parameters

        return validate_parameters(parameters)
    if check == 'native_run_context':
        from .native_context import validate_parameters

        return validate_parameters(parameters)
    if check == 'trajectory_sampling' and 'sample_times' not in parameters:
        raise InputError('Trajectory sampling cases require explicit sample_times')
    if check == 'resolved_layout' and not parameters.get('expected_names'):
        raise InputError('Resolved-layout cases require an explicit expected BODY-name order')
    if check == 'act_query_cadence':
        limit = parameters.get('max_query_gap')
        if type(limit) is not int or not 1 <= limit <= 1000000:
            raise InputError('max_query_gap must be an explicit integer in [1, 1000000]')
    if check == 'native_named_action' and any(name in parameters and type(parameters[name]) is not bool for name in
            ('require_canonical_matches_driver_return', 'require_recorded_matches_driver_return')):
        raise InputError('Named equality requirement must be a strict Boolean')
    try:
        result = deepcopy(parameters)
        if check in ('act_query_cadence', 'act_command_path', 'native_named_action'):
            result.setdefault('atol', 1e-6)
            result.setdefault('rtol', 1e-5)
        finite_tree(result)
        _encode(result)
    except (TypeError, ValueError, RecursionError) as exc:
        raise InputError('Incident parameters must be bounded finite declarative values') from exc
    return result


def _controls(check, controls):
    if controls is None:
        return []
    if not isinstance(controls, list) or len(controls) > 4:
        raise InputError('Require at most four named healthy/unknown controls')
    result, names = [], set()
    for control in controls:
        if not isinstance(control, dict) or set(control) != {'name', 'source', 'parameters', 'expected_status'}:
            raise InputError('Control requires name, source, parameters and expected_status')
        name = control['name']
        if (not isinstance(name, str) or not 1 <= len(name) <= 128 or name != name.strip() or
                not name.isprintable() or name in names):
            raise InputError('Require unique bounded printable control labels without surrounding whitespace')
        if control['expected_status'] not in ('pass', 'warning', 'unknown'):
            raise InputError('A control must explicitly expect pass, warning or unknown')
        names.add(name)
        result.append({'name': name, 'source': control['source'], 'parameters': _parameters(check, control['parameters']),
                       'expected_status': control['expected_status']})
    return result


def _copy_source(source, target, check, case_root):
    if check == 'native_run_context':
        from .native_evidence import snapshot_native_context

        target.parent.mkdir(parents=True, exist_ok=True)
        copied = snapshot_native_context(source, target)
        _inventory(case_root)
        return copied
    if check == 'trajectory_sampling':
        snapshot = {source.name: _hash(source)}
        sources = [(source, target / ('trajectory' + source.suffix.lower()))]
    else:
        snapshot = _inventory(source)
        sources = [(source / name, target / name) for name in sorted(snapshot)]
    existing = _inventory(case_root)
    if (len(existing) + len(snapshot) >= MAX_PAYLOAD_FILES or
            sum(v[1] for v in existing.values()) + sum(v[1] for v in snapshot.values()) > MAX_TOTAL_BYTES):
        raise InputError('Complete native input trees exceed incident case caps; use a small shareable incident (media is not silently omitted)')
    target.mkdir(parents=True)
    for original, destination in sources:
        expected, size = snapshot[original.name if check == 'trajectory_sampling' else str(original.relative_to(source))]
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest, copied = hashlib.sha256(), 0
        try:
            with original.open('rb') as stream, destination.open('xb') as output:
                while chunk := stream.read(512 * 1024):
                    copied += len(chunk)
                    if copied > size or copied > MAX_FILE_BYTES:
                        raise InputError('Incident source changed while copying')
                    digest.update(chunk)
                    output.write(chunk)
        except OSError as exc:
            raise InputError(f'Cannot snapshot incident evidence: {exc}') from exc
        if copied != size or digest.hexdigest() != expected or _hash(original) != (expected, size):
            raise InputError('Incident source changed while copying')
    current = {source.name: _hash(source)} if check == 'trajectory_sampling' else _inventory(source)
    if current != snapshot:
        raise InputError('Incident source tree changed while copying')
    return sources[0][1] if check == 'trajectory_sampling' else target


def _run(check, source, parameters, *, reference=None):
    if check == 'native_observation_source':
        from .native_observation_source import audit_native_observation_source

        return audit_native_observation_source(source, **parameters)
    if check == 'native_processor_source':
        from .native_processor_source import audit_native_processor_source

        return audit_native_processor_source(source, **parameters)
    if check == 'evaluation_cohort':
        from .evaluation_cohort import audit_evaluation_cohort

        return audit_evaluation_cohort(str(source), reference=str(reference) if reference is not None else None, **parameters)
    if check == 'native_run_context':
        from .native_context import audit_native_run_context

        return audit_native_run_context(str(source), reference=str(reference) if reference is not None else None, **parameters)
    if check == 'native_named_action':
        from .named_actions import audit_named_actions

        return audit_named_actions(str(source), **parameters)
    if check == 'act_command_path':
        from .command_path import audit_command_capture

        return audit_command_capture(str(source), **parameters)
    if check == 'act_query_cadence':
        from .cadence import audit_query_cadence

        return audit_query_cadence(str(source), **parameters)
    if check == 'episode_windows':
        return audit_episode_windows(str(source), **parameters)
    if check == 'resolved_layout':
        from .layouts import audit_layout_capture

        return audit_layout_capture(str(source), **parameters)
    return sample_joint_trajectory(str(source), **parameters)


def _patches(report, before):
    proposals = report.get('metadata_patches', [])
    if (report['status'] != 'error' or not report['coverage'].get('identity_verified') or
            not report['coverage'].get('complete') or report.get('metadata_patches_truncated') or
            not proposals or report['counts']['metadata_patches'] != len(proposals)):
        raise InputError('Metadata correction requires complete, unambiguous, untruncated fresh proposals')
    result, seen = {}, set()
    for proposal in proposals:
        path = Path(proposal['path'])
        if not path.is_relative_to(before):
            raise InputError('Metadata proposal escapes the staged before dataset')
        relative = str(path.relative_to(before))
        if not re.fullmatch(r'meta/episodes/chunk-[0-9]{3,}/file-[0-9]{3,}\.parquet', relative):
            raise InputError('Only native episode metadata cells can be corrected')
        column, row = proposal.get('column'), proposal.get('row')
        if (column not in ('dataset_from_index', 'dataset_to_index', 'length') or type(row) is not int or row < 0 or
                any(type(proposal.get(k)) is not int or not 0 <= proposal[k] < 2**53 for k in ('observed', 'expected')) or
                (relative, row, column) in seen):
            raise InputError('Invalid or ambiguous native metadata correction proposal')
        if _hash(_inside(before, relative))[0] != proposal['sha256']:
            raise InputError('Native metadata proposal hash is stale')
        seen.add((relative, row, column))
        result.setdefault(relative, []).append(proposal)
    return result


def _metadata_table(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise InputError('Install policy-deployment-doctor[dataset] for metadata copy correction') from exc
    with path.open('rb') as stream:
        stream.seek(-8, 2)
        tail = stream.read(8)
    if tail[4:] != b'PAR1' or not 0 < int.from_bytes(tail[:4], 'little') <= 2 * 1024 * 1024:
        raise InputError('Metadata correction footer budget exceeded')
    from .native_parquet_guard import require_unencrypted_parquet

    require_unencrypted_parquet(path)
    parquet = pq.ParquetFile(path)
    meta = parquet.metadata
    if meta.num_rows > 100000 or meta.num_columns > 128 or meta.num_row_groups > 512:
        raise InputError('Metadata correction table budget exceeded')
    expanded = values = 0
    for group in range(meta.num_row_groups):
        for column in range(meta.num_columns):
            field = meta.row_group(group).column(column)
            expanded += field.total_uncompressed_size
            values += field.num_values
    if expanded > 32 * 1024 * 1024 or values > 1000000:
        raise InputError('Metadata correction expanded-byte/value budget exceeded')
    return parquet.read()


def _proposed_table(before_table, proposals):
    import pyarrow as pa
    result = before_table
    for proposal in proposals:
        column, row = proposal['column'], proposal['row']
        position = before_table.schema.get_field_index(column)
        if position < 0 or row >= before_table.num_rows or not pa.types.is_integer(before_table.schema.field(column).type):
            raise InputError('Metadata correction target is not a native integer cell')
        if before_table.column(column)[row].as_py() != proposal['observed']:
            raise InputError('Metadata correction observed cell is stale')
        values = result.column(column).to_pylist()
        values[row] = proposal['expected']
        result = result.set_column(position, before_table.schema.field(column), pa.array(values, type=before_table.schema.field(column).type))
    return result


def _metadata_changes(before, after, report, *, apply=False):
    proposals = _patches(report, before)
    original_files, corrected_files = _inventory(before), _inventory(after)
    if original_files.keys() != corrected_files.keys():
        raise InputError('Corrected dataset must retain the exact native file set')
    for name in original_files:
        if name not in proposals and original_files[name] != corrected_files[name]:
            raise InputError('A non-target native dataset file changed')
    changed, cells = [], []
    for name, changes in proposals.items():
        original = _inside(before, name)
        corrected = _inside(after, name)
        before_table = _metadata_table(original)
        proposed = _proposed_table(before_table, changes)
        if apply:
            import pyarrow.parquet as pq
            pq.write_table(proposed, corrected)
        actual = _metadata_table(corrected)
        if not proposed.equals(actual, check_metadata=True):
            raise InputError('Corrected metadata changes schema or non-target logical cells')
        changed.append({'path': str(corrected), 'before_sha256': original_files[name][0], 'after_sha256': _hash(corrected)[0]})
        cells.extend({'path': str(corrected), 'row': p['row'], 'column': p['column'],
                      'observed': p['observed'], 'expected': p['expected']} for p in changes)
    return {'method': 'exact_native_metadata_copy', 'logical_cells_changed': len(cells), 'schema_preserved': True,
            'other_logical_cells_unchanged': True, 'non_target_files_unchanged': True, 'changed_files': changed,
            'cells': cells, 'scope': 'Native metadata-only correction on a separate copy; no physical repair claim'}


def _retained_run(check, source, parameters, reference):
    """Pass a reference only to the family that consumes one."""
    if check in ('native_run_context', 'evaluation_cohort'):
        return _run(check, source, parameters, reference=reference)
    return _run(check, source, parameters)


def _evaluate(root, check, before_name, after_name, parameters, controls):
    before_inventory = _inventory(root, case=(root / 'manifest.json').exists())
    before, after = _inside(root, before_name), _inside(root, after_name)
    retained_reference = after if check in ('native_run_context', 'evaluation_cohort') else None
    report = {'before': _retained_run(check, before, parameters, retained_reference),
              'after': _retained_run(check, after, parameters, retained_reference), 'controls': []}
    if check == 'native_observation_source':
        changes = {'method': 'supplied_native_observation_source_capture', 'same_check_parameters': True,
            'scope': 'Declared pre-normalization state/camera API values checked against retained original-derived inputs. '
                     'This evidence-only check does not rerun native code, authenticate original origin or establish '
                     'physical camera identity, training recipe, policy competence or a physical repair.'}
    elif check == 'native_processor_source':
        changes = {'method': 'supplied_native_processor_source_capture', 'same_check_parameters': True,
            'scope': 'Explicit caller restoration intent checked against witnessed source bytes and a same-input native reference. '
                     'Derived views can share one native attempt. This evidence-only check does not rerun native preprocessing '
                     'or establish a loader defect, policy competence, physical repair or authenticated source origin.'}
    elif check == 'native_run_context':
        changes = {'method': 'supplied_recorded_context_reference', 'same_check_parameters': True,
            'reference': 'after/env_cfg.json',
            'scope': 'The supplied reference replaces a comparison selection that violates selected recorded-setting invariants. '
                     'Numeric excerpts retain native paths/dtypes/all selected rows; media and other omissions are listed in evidence_selection.json. '
                     'This establishes no behavioral correction, runtime/asset equivalence or independent origin authenticity.'}
    elif check == 'evaluation_cohort':
        changes = {'method': 'supplied_evaluation_context_reference', 'same_check_parameters': True,
            'reference': 'after/',
            'scope': 'The supplied accepted-context reference and observed episode keys are retained. '
                     'Selecting this reference is not a demonstrated behavioral correction. '
                     'Declared counts, shards, native summaries, physical pairing and causal superiority remain unverified.'}
    elif check == 'episode_windows':
        changes = _metadata_changes(before, after, report['before'])
    elif check == 'native_named_action':
        changes = {'method': 'supplied_native_named_action', 'same_check_parameters': True,
            'scope': 'Supplied named software-boundary correction under explicit order/namespace/optional equality criteria. '
                     'Lawful limiting is not intrinsically a fault; no recorded action or physical repair is inferred.'}
    elif check == 'act_command_path':
        changes = {'method': 'supplied_native_command_path', 'same_check_parameters': True,
            'scope': 'Separately supplied native software-consumer captures under the corrected normative processor/order/namespace contract. '
                     'This does not infer named-joint mapping, recorded action or physical repair.'}
    elif check == 'act_query_cadence':
        changes = {'method': 'supplied_act_query_cadence_capture', 'same_check_parameters': True,
                   'scope': 'Separate supplied native executions under the same explicit selection-call query requirement; '
                            'no policy-quality, lineage, downstream or physical repair claim.'}
    elif check == 'resolved_layout':
        a, b = report['before']['layout'], report['after']['layout']
        if (not a.get('sensor_identity') or a['sensor_identity'] != b.get('sensor_identity') or
                not a.get('body_names') or set(a['body_names']) != set(b.get('body_names') or [])):
            raise InputError('Corrected layout needs the same evidenced sensor namespace and selected body set')
        changes = {'method': 'supplied_resolved_layout_capture', 'same_sensor_namespace': True,
                   'same_selected_body_set': True, 'same_check_parameters': True,
                   'scope': 'Supplied raw-term snapshot under the retained layout contract; '
                            'namespace equality does not establish model or physical equivalence.'}
    else:
        if set(report['before']['joint_names']) != set(report['after']['joint_names']):
            raise InputError('Corrected trajectory must retain the native joint set')
        changes = {'method': 'supplied_native_trajectory', 'same_native_joint_set': True, 'same_check_parameters': True,
                   'before_sha256': _hash(before)[0], 'after_sha256': _hash(after)[0],
                   'scope': 'Supplied corrected native artifact under the same sampled checks; no physical repair claim'}
    for control in controls:
        if check in ('native_processor_source', 'native_observation_source') and control['parameters'] != parameters:
            raise InputError('Processor-source controls must retain the same immutable source criterion')
        if check in ('native_run_context', 'evaluation_cohort') and control['parameters'] != parameters:
            raise InputError('Recorded-context controls must retain identical selected invariants')
        if check == 'native_named_action' and control['parameters'] != parameters:
            raise InputError('Named-action controls must retain identical criteria and tolerances')
        if check == 'act_command_path' and control['parameters'] != parameters:
            raise InputError('Command-path controls must retain the same processor/order/namespace criteria and tolerances')
        if check == 'act_query_cadence' and control['parameters'] != parameters:
            raise InputError('Cadence controls must retain the same explicit requirement and tolerances')
        check_report = _retained_run(check, _inside(root, control['source']), control['parameters'], retained_reference)
        report['controls'].append({'name': control['name'], 'expected_status': control['expected_status'], 'check': check_report})
    report['change_verification'] = changes
    if _inventory(root, case=(root / 'manifest.json').exists()) != before_inventory:
        raise InputError('Incident evidence changed during diagnostic evaluation')
    paths = {str((root / name).resolve()): name for name in before_inventory}
    # Change receipts carry a hash pair rather than an artifact sha256.
    for cell in changes.get('cells', []) + changes.get('changed_files', []):
        cell['path'] = str(Path(cell['path']).relative_to(root))
    return _portable(report, paths)


def _lifecycle(report):
    if report['before'].get('kind') in ('native_processor_source_audit', 'native_observation_source_audit'):
        before, after = report['before'], report['after']
        if (before['source_criterion']['status'] != 'error'
                or before['source_criterion']['code'] != 'source_contradiction'
                or after['source_criterion']['status'] != 'pass'
                or not isinstance(before.get('input_digest'), str) or len(before['input_digest']) != 64
                or before['input_digest'] != after.get('input_digest')
                or not before.get('native_identity') or before['native_identity'] != after.get('native_identity')):
            return False
        if before.get('kind') == 'native_observation_source_audit' and (
                not before.get('model_identity') or before['model_identity'] != after.get('model_identity')):
            return False
    if report['before'].get('kind') in ('command_path_audit', 'named_action_audit') and (
            report['before']['acceptance_contract']['status'] != 'error' or
            report['after']['acceptance_contract']['status'] != 'pass'):
        return False
    if report['before'].get('kind') == 'query_cadence_audit' and (
            report['before']['cadence_criterion']['status'] != 'error' or
            report['after']['cadence_criterion']['status'] != 'pass'):
        return False
    # A corrected window audit may keep advisory warnings (for example a single-frame episode); the error must be gone.
    after_ok = report['after']['status'] == 'pass' or (report['after'].get('kind') == 'episode_window_audit'
                                                       and report['after']['status'] == 'warning')
    return (report['before']['status'] == 'error' and after_ok and
            all(c['check']['status'] == c['expected_status'] for c in report['controls']))


def _summary(report):
    if report.get('kind') in ('native_run_context_audit', 'evaluation_cohort_audit', 'native_processor_source_audit', 'native_observation_source_audit'):
        return report
    if report.get('kind') == 'named_action_audit':
        return {key: report[key] for key in ('kind', 'status', 'binding_status', 'numeric_replay_status',
            'acceptance_contract', 'action_names', 'engine_action_names', 'driver_namespace', 'tick_count',
            'driver_change_events', 'canonical_mutation_events', 'first_driver_change', 'robot_processor_semantics_status', 'first_intrinsic_contradiction',
            'recorded_action_status', 'recording', 'physical_execution_status', 'policy_selection_status', 'policy_processor_status',
            'artifacts', 'scope')}
    if report.get('kind') == 'command_path_audit':
        return {key: report[key] for key in ('kind', 'status', 'binding_status', 'numeric_replay_status',
            'acceptance_contract', 'observed_processor', 'action_names', 'consumer_namespace', 'tick_count',
            'query_lineage_status', 'recorded_action_status', 'physical_execution_status', 'artifacts', 'scope')}
    if report.get('kind') == 'query_cadence_audit':
        return {key: report[key] for key in ('kind', 'status', 'consumer_requirement', 'cadence_criterion',
                'numeric_replay_status', 'first_numeric_contradiction', 'query_lineage_status', 'downstream_association',
                'source_supported', 'coverage_supported', 'artifacts', 'tolerances', 'scope')}
    if report.get('kind') == 'resolved_layout_audit':
        return {key: report[key] for key in ('kind', 'status', 'counts', 'raw_binding', 'processed_binding',
                'layout', 'acceptance_contract', 'artifacts', 'scope')} | {
                'findings': [finding for finding in report['findings'] if finding['status'] != 'pass'][:4]}
    if report.get('kind') == 'episode_window_audit':
        result = {k: report[k] for k in ('kind', 'status', 'counts', 'coverage')}
        result.update(findings=report['findings'][:4], metadata_patches=report['metadata_patches'][:4], query=None,
                      findings_truncated=report['counts']['findings'] > 4,
                      metadata_patches_truncated=report['counts']['metadata_patches'] > 4,
                      artifacts=report['artifacts'][:4], artifact_count=len(report['artifacts']),
                      artifacts_truncated=len(report['artifacts']) > 4)
        if report['query'] is not None:
            query = report['query']
            selected = {k: query[k] for k in ('episode_index', 'anchor_index', 'delta_indices')}
            for name in ('declared', 'corrected_what_if'):
                window = query[name]
                if 'indices' not in window:
                    selected[name] = window
                    continue
                values = window['values']
                selected[name] = {k: window[k] for k in ('status', 'indices', 'action_is_pad', 'selected_episode_indices', 'value_status')}
                selected[name].update(reason=window.get('reason'), values=None if values is None else values[:8], value_rows=None if values is None else len(values),
                    values_truncated=values is not None and len(values) > 8,
                    value_reason=window.get('value_reason'),
                    cross_episode_witness_count=len(window['cross_episode_witnesses']),
                    cross_episode_witnesses=window['cross_episode_witnesses'][:4],
                    cross_episode_witnesses_truncated=len(window['cross_episode_witnesses']) > 4)
            result['query'] = selected
        return result
    return {k: report[k] for k in ('kind', 'status', 'sampling_complete', 'interpolation_method', 'joint_names',
            'execution_status', 'physical_cause')} | {'samples': report['samples'][:8], 'sample_count': len(report['samples']),
            'samples_truncated': len(report['samples']) > 8, 'sampled_limit_violation_count': len(report['sampled_limit_violations']),
            'sampled_limit_violations': report['sampled_limit_violations'][:4],
            'sampled_limit_violations_truncated': len(report['sampled_limit_violations']) > 4,
            'artifacts': report['artifacts'], 'excursion_count': report['excursion_count']}


def _checks(report):
    return {'before': _summary(report['before']), 'after': _summary(report['after']),
            'controls': [{'name': c['name'], 'expected_status': c['expected_status'], 'check': _summary(c['check'])}
                         for c in report['controls']]}


def _fingerprint(check, yaml_used):
    names = ('incidents.py', 'cases.py', 'windows.py', 'trajectory.py', 'artifacts.py', 'publication.py')
    if check == 'native_processor_source':
        names += ('native_processor_source.py',)
    if check == 'native_observation_source':
        names += ('native_observation_source.py', 'rollouts.py')
    if check == 'resolved_layout':
        names += ('layouts.py', 'rollouts.py')
    if check == 'act_query_cadence':
        names += ('cadence.py', 'selection.py', 'selection_bindings.py', 'layouts.py', 'rollouts.py')
    if check == 'native_named_action':
        names += ('named_actions.py', 'named_rules.py', 'command_rules.py', 'selection.py', 'layouts.py', 'rollouts.py')
    if check == 'act_command_path':
        names += ('command_path.py', 'command_rules.py', 'selection.py', 'layouts.py', 'rollouts.py')
    if check == 'evaluation_cohort':
        names += ('evaluation_cohort.py',)
    if check == 'native_run_context':
        names += ('native_context.py', 'native_evidence.py')
    root = Path(__file__).parent
    result = {'engine_version': __version__, 'module_sha256': {name: _digest((root / name).read_bytes()) for name in names},
              'numpy_version': importlib.metadata.version('numpy')}
    if check == 'episode_windows':
        result['pyarrow_version'] = importlib.metadata.version('pyarrow')
    if check == 'native_run_context':
        result['h5py_version'] = importlib.metadata.version('h5py')
    if yaml_used:
        result['pyyaml_version'] = importlib.metadata.version('PyYAML')
    return result


def _response_bound(report):
    if len(json.dumps(report, allow_nan=False).encode()) > 192 * 1024:
        raise InputError('Incident response exceeds the 192 KiB budget; narrow the incident')
    return report


def export_incident_case(check, source, destination, parameters, *, corrected=None,
                         apply_metadata_corrections=False, controls=None):
    if not isinstance(check, str) or check not in CHECKS or type(apply_metadata_corrections) is not bool:
        raise InputError('Require an allowlisted check and a strict Boolean metadata-copy option')
    if (corrected is None) == (not apply_metadata_corrections) or (apply_metadata_corrections and check != 'episode_windows'):
        raise InputError('Choose supplied corrected evidence or explicit windows-only metadata copy correction')
    if check == 'native_observation_source' and type(parameters) is dict and not parameters:
        from .native_observation_source import bind_native_observation_source_parameters

        parameters = bind_native_observation_source_parameters(_native_source(corrected, check))
        if type(controls) is list:
            controls = deepcopy(controls)
            for control in controls:
                if type(control) is dict and control.get('parameters') == {}:
                    control['parameters'] = deepcopy(parameters)
    parameters, controls = _parameters(check, parameters), _controls(check, controls)
    if not isinstance(destination, (str, Path)) or not str(destination):
        raise InputError('Require a new incident destination path')
    source = _native_source(source, check)
    corrected_source = _native_source(corrected, check) if corrected is not None else None
    if check in ('native_processor_source', 'native_observation_source'):
        if check == 'native_observation_source':
            from .native_observation_source import bind_native_observation_source_parameters as bind_source
        else:
            from .native_processor_source import bind_native_processor_source_parameters as bind_source

        initial_parameters = deepcopy(parameters)
        parameters = bind_source(corrected_source, parameters)
        for control in controls:
            if control['parameters'] != initial_parameters:
                raise InputError('Processor-source controls must retain the same source criterion')
            control['parameters'] = deepcopy(parameters)
    if check == 'native_named_action':
        from .named_actions import audit_named_actions

        healthy = audit_named_actions(str(corrected_source))
        if healthy['binding_status'] != 'pass' or healthy['numeric_replay_status'] != 'pass':
            raise InputError('Corrected named capture requires a complete supported native boundary')
        parameters.setdefault('expected_action_names', deepcopy(healthy['action_names']))
        parameters.setdefault('expected_driver_namespace', healthy['driver_namespace'])
        parameters.setdefault('require_canonical_matches_driver_return', False)
        parameters.setdefault('require_recorded_matches_driver_return', False)
        for control in controls:
            for name, value in parameters.items():
                control['parameters'].setdefault(name, deepcopy(value))
    if check == 'act_command_path':
        from .command_path import audit_command_capture

        healthy = audit_command_capture(str(corrected_source))
        if healthy['binding_status'] != 'pass' or healthy['numeric_replay_status'] != 'pass' or healthy['observed_processor'] is None:
            raise InputError('Corrected command capture needs a complete supported chain and uniform actual processor statistics')
        for name, value in (('expected_processor', healthy['observed_processor']),
                            ('expected_action_names', healthy['action_names']),
                            ('expected_consumer_namespace', healthy['consumer_namespace'])):
            if parameters.get(name) is None:
                parameters[name] = deepcopy(value)
        for control in controls:
            for name, value in parameters.items():
                control['parameters'].setdefault(name, deepcopy(value))
    if check == 'resolved_layout':
        # Retain effective semantics from the reviewed healthy snapshot instead of
        # requiring a caller-written contract file. The expected body order stays explicit.
        from .layouts import audit_layout_capture

        healthy = audit_layout_capture(str(corrected_source))
        parameters.setdefault('atol', 1e-6)
        parameters.setdefault('rtol', 1e-5)
        for name, field in (('expected_frame', 'frame'), ('expected_quantity', 'quantity'),
                            ('expected_identity', 'sensor_identity')):
            if parameters.get(name) is None:
                value = healthy['layout'].get(field)
                if not value:
                    raise InputError('Corrected layout lacks evidenced frame, quantity or sensor namespace')
                parameters[name] = deepcopy(value)
    control_sources = [_native_source(c['source'], check) for c in controls]
    target = Path(destination).expanduser().absolute()
    target = target.parent.resolve() / target.name
    if target.exists() or target.is_symlink():
        raise InputError('Incident destination already exists; choose a new directory')
    for original in [source, corrected_source, *control_sources]:
        if original is not None and (target == original or (original.is_dir() and target.is_relative_to(original))):
            raise InputError('Incident destination cannot be nested inside a native input tree')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.policy-doctor-incident-', dir=target.parent))
    try:
        before = _copy_source(source, staging / 'before', check, staging)
        after = _copy_source(corrected_source, staging / 'after', check, staging) if check in ('native_run_context', 'evaluation_cohort') else None
        primary_before = _retained_run(check, before, parameters, after)
        if check in ('native_processor_source', 'native_observation_source') and primary_before['source_criterion']['code'] != 'source_contradiction':
            raise InputError('Processor-source cases require a positive witnessed source contradiction before correction')
        if check in ('act_command_path', 'native_named_action') and primary_before['acceptance_contract']['status'] != 'error':
            raise InputError('Command-path cases require an observed processor/order/namespace contract violation, not numeric error alone')
        if check == 'act_query_cadence' and primary_before['cadence_criterion']['status'] != 'error':
            raise InputError('Cadence cases require an observed requirement violation before correction')
        if primary_before['status'] != 'error':
            raise InputError('A primary incident must retain a supported diagnostic error before correction')
        if after is None:
            after = _copy_source(before if corrected_source is None else corrected_source, staging / 'after', check, staging)
        if apply_metadata_corrections:
            _metadata_changes(before, after, primary_before, apply=True)
        native_controls = []
        for index, (control, original) in enumerate(zip(controls, control_sources, strict=True)):
            copied = _copy_source(original, staging / 'controls' / str(index), check, staging)
            native_controls.append(dict(control, source=str(copied.relative_to(staging))))
        before_name, after_name = str(before.relative_to(staging)), str(after.relative_to(staging))
        observed = _evaluate(staging, check, before_name, after_name, parameters, native_controls)
        if not _lifecycle(observed):
            raise InputError('Corrected evidence must pass the same parameters and all named control statuses must match')
        inventory = _inventory(staging)
        expected = _encode(observed)
        (staging / 'expected.json').write_bytes(expected)
        files = {name: value[0] for name, value in sorted(inventory.items())} | {'expected.json': _digest(expected)}
        yaml_used = check == 'trajectory_sampling' and any(name.endswith(('.yaml', '.yml')) for name in files)
        verifier = _fingerprint(check, yaml_used)
        manifest = {'schema_version': 1, 'kind': KIND, 'check': check, 'parameters': parameters,
                    'before': before_name, 'after': after_name, 'controls': native_controls,
                    'files': files, 'verifier': verifier}
        encoded = _encode(manifest)
        (staging / 'manifest.json').write_bytes(encoded)
        (staging / 'README.md').write_text('# Native robot incident case\n\n'
            'This case retains an error, a corrected native artifact and optional healthy/unknown controls. '
            'Install the engine/dependency versions recorded in manifest.json, then run:\n\n'
            '```sh\npolicy-doctor reproduce-case .\n```\n\n'
            'The read-only MCP reproduce_incident_case tool runs the same allowlisted checks. '
            'expected.json contains full findings; a reproduced case preserves its before error, its corrected outcome (pass, or warning for advisory window findings) and its control statuses. '
            'No original author paths or artifact-supplied code are required. A corrected numeric check is not a physical repair.\n\n'
            'Window incidents retain the complete bounded native input trees; media is not silently omitted. '
            'Limits are 128 MiB per file, 256 MiB total case evidence, 256 payload files and directory depth 8. '
            'This is a small shareable-incident workflow, not general packaging of large training datasets.\n')
        if check == 'native_run_context':
            (staging / 'README.md').write_text('# Recorded evaluation comparison case\n\n'
                'Run `policy-doctor reproduce-case .`, then `policy-doctor check-case . LATER_RUN`. '
                'The same selected JSON pointers are compared with the retained reference in after/. '
                'The before error concerns comparison eligibility, not policy failure; the reference is not a demonstrated physical fix.\n\n'
                'env_cfg.json and bounded logs/license files retain original bytes. HDF5 files are derived numeric excerpts '
                'with original native dataset paths, dtypes and all rows of selected arrays. '
                'Each evidence_selection.json records the original full-file hashes, retained values and explicit omissions. '
                'Images are omitted without decoding. Source association is an author assertion; without the originals '
                'a colleague verifies retained integrity and repeatable checks, not independent original membership. '
                'Check the source licenses before redistribution. Missing effective runtime/code/asset evidence remains unknown.\n')
        if check == 'evaluation_cohort':
            (staging / 'README.md').write_text('# Observed evaluation comparison case\n\n'
                'Run `policy-doctor reproduce-case .`, then `policy-doctor check-case . LATER_RUN`. '
                'The same native JSON selectors and selected context are compared against after/. '
                'Native input files retain their original bytes. The before error concerns comparison selection; '
                'the accepted-context reference is not a physical fix. Declared counts, summaries, shards, '
                'physical initial states and causal performance remain unverified. Check source licenses before redistribution.\n')
        (staging / 'READY.json').write_bytes(_encode({'schema_version': 1, 'manifest_sha256': _digest(encoded)}))
        # Evaluate a stable published candidate, and reject an oversized response before publication.
        if {name: value[0] for name, value in _inventory(staging, case=True).items() if name not in ('manifest.json', 'READY.json', 'README.md')} != files:
            raise InputError('Incident evidence changed while preparing the case')
        response = _response_bound({'schema_version': 1, 'kind': 'exported_incident_case', 'status': 'pass',
            'before_status': 'error', 'after_status': observed['after']['status'], 'directory': str(target), 'checks': _checks(observed),
            'change_verification': observed['change_verification'], 'verifier': verifier,
            'files': dict(list(files.items())[:16]), 'file_count': len(files), 'files_truncated': len(files) > 16, 'scope': SCOPE})
        publish_capture(staging, target)
        return response
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _manifest(root):
    inventory = _inventory(root, case=True)
    if not {'manifest.json', 'READY.json', 'expected.json', 'README.md'} <= set(inventory):
        raise InputError('Incident case is incomplete')
    manifest, ready = read_json(root / 'manifest.json'), read_json(root / 'READY.json')
    if (set(ready) != {'schema_version', 'manifest_sha256'} or type(ready['schema_version']) is not int or
            ready['schema_version'] != 1 or ready['manifest_sha256'] != inventory['manifest.json'][0]):
        raise InputError('Incident commit marker or manifest hash mismatch')
    if (set(manifest) != {'schema_version', 'kind', 'check', 'parameters', 'before', 'after', 'controls', 'files', 'verifier'} or
            type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or manifest['kind'] != KIND or
            not isinstance(manifest['check'], str) or manifest['check'] not in CHECKS):
        raise InputError('Unsupported declarative incident manifest')
    check = manifest['check']
    _parameters(check, manifest['parameters'])
    if check == 'resolved_layout' and any(not manifest['parameters'].get(name) for name in
            ('expected_frame', 'expected_quantity', 'expected_identity')):
        raise InputError('A retained resolved-layout case requires its frame, quantity and sensor namespace criteria')
    if check == 'resolved_layout' and not {'atol', 'rtol'} <= set(manifest['parameters']):
        raise InputError('A retained resolved-layout case requires explicit numeric tolerances')
    if check == 'act_query_cadence' and not {'atol', 'rtol'} <= set(manifest['parameters']):
        raise InputError('A retained cadence case requires explicit numeric tolerances')
    if check == 'act_command_path' and any(manifest['parameters'].get(n) is None for n in
            ('expected_processor', 'expected_action_names', 'expected_consumer_namespace', 'atol', 'rtol')):
        raise InputError('Retained command-path criteria must include processor, declared order, namespace and explicit tolerances')
    if check == 'native_named_action' and (any(manifest['parameters'].get(n) is None for n in
            ('expected_action_names', 'expected_driver_namespace', 'atol', 'rtol'))
            or type(manifest['parameters'].get('require_canonical_matches_driver_return')) is not bool
            or type(manifest['parameters'].get('require_recorded_matches_driver_return', False)) is not bool):
        raise InputError('Retained named-action case requires full criteria and a strict Boolean equality requirement')
    controls = _controls(check, manifest['controls'])
    if check in ('native_run_context', 'evaluation_cohort') and any(c['parameters'] != manifest['parameters'] for c in controls):
        raise InputError('Recorded-context controls must retain identical selected invariants')
    if check == 'native_named_action' and any(c['parameters'] != manifest['parameters'] for c in controls):
        raise InputError('Retained named-action controls must use identical criteria')
    if check == 'act_command_path' and any(c['parameters'] != manifest['parameters'] for c in controls):
        raise InputError('Retained command-path controls must use identical criteria')
    if check == 'act_query_cadence' and any(c['parameters'] != manifest['parameters'] for c in controls):
        raise InputError('Cadence controls must retain the same criteria')
    for role in ('before', 'after'):
        name = manifest[role]
        expected = [role + '/trajectory' + suffix for suffix in ('.json', '.yaml', '.yml')] if check == 'trajectory_sampling' else role
        if (name != expected if isinstance(expected, str) else name not in expected):
            raise InputError('Invalid fixed native incident source locator')
        _inside(root, name)
    for index, control in enumerate(controls):
        prefix = 'controls/' + str(index)
        if (control['source'] != prefix if check != 'trajectory_sampling' else
                control['source'] not in [prefix + '/trajectory' + s for s in ('.json', '.yaml', '.yml')]):
            raise InputError('Invalid fixed native incident control locator')
        _inside(root, control['source'])
    files = manifest['files']
    if not isinstance(files, dict) or 'expected.json' not in files or set(inventory) != set(files) | {'manifest.json', 'READY.json', 'README.md'}:
        raise InputError('Invalid or extra incident payload files')
    for name, expected in files.items():
        _inside(root, name)
        sources = [manifest['before'], manifest['after'], *(c['source'] for c in controls)]
        if name != 'expected.json' and not any(name == s if check == 'trajectory_sampling' else name.startswith(s + '/') for s in sources):
            raise InputError('Unexpected incident payload outside the declared native inputs')
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected) or inventory[name][0] != expected:
            raise InputError('Incident payload hash mismatch')
    return manifest, inventory


def reproduce_incident_case(directory):
    if not isinstance(directory, (str, Path)) or not str(directory):
        raise InputError('Require an incident directory path')
    original = Path(directory).expanduser()
    if original.is_symlink():
        raise InputError('Incident case directory symlink refused')
    root = original.resolve()
    if not root.is_dir():
        raise InputError('Expected an incident case directory')
    manifest, before_inventory = _manifest(root)
    observed = _evaluate(root, manifest['check'], manifest['before'], manifest['after'], manifest['parameters'], manifest['controls'])
    expected = read_json(root / 'expected.json')
    matches = _encode(observed) == _encode(expected)
    yaml_used = manifest['check'] == 'trajectory_sampling' and any(name.endswith(('.yaml', '.yml')) for name in manifest['files'])
    current = _fingerprint(manifest['check'], yaml_used)
    verifier_matches = current == manifest['verifier']
    if _inventory(root, case=True) != before_inventory:
        raise InputError('Incident case changed while it was being checked')
    lifecycle = _lifecycle(observed)
    files = manifest['files']
    return _response_bound({'schema_version': 1, 'kind': 'incident_case_reproduction',
        'status': 'unknown' if not verifier_matches else 'error' if not matches or not lifecycle else 'pass',
        'reproduction_matches': matches, 'verifier_matches': verifier_matches, 'lifecycle_verified': lifecycle,
        'recorded_verifier': manifest['verifier'], 'current_verifier': current, 'checks': _checks(observed),
        'change_verification': observed['change_verification'], 'checked_files': dict(list(files.items())[:16]),
        'checked_file_count': len(files), 'checked_files_truncated': len(files) > 16,
        'recorded_findings_sha256': _digest(_encode(expected)), 'current_findings_sha256': _digest(_encode(observed)), 'scope': SCOPE})


def check_incident_candidate(directory, candidate):
    """Apply a verified case's frozen check to later native evidence, without writes."""
    if not isinstance(directory, (str, Path)) or not str(directory):
        raise InputError('Require an incident case directory')
    original = Path(directory).expanduser()
    if original.is_symlink():
        raise InputError('Incident case directory symlink refused')
    root = original.resolve()
    case_inventory = _inventory(root, case=True)
    verification = reproduce_incident_case(str(root))
    if _inventory(root, case=True) != case_inventory:
        raise InputError('Incident case changed while establishing candidate prerequisites')
    result = {'schema_version': 1, 'kind': 'incident_candidate_check', 'status': verification['status'],
        'case_status': verification['status'], 'case_verified': False, 'candidate_evaluated': False,
        'expected_status': 'pass', 'candidate': None, 'parameters': None, 'check': None,
        'case_verification': {k: verification[k] for k in ('reproduction_matches', 'verifier_matches', 'lifecycle_verified',
            'recorded_findings_sha256', 'current_findings_sha256')},
        'scope': 'Runs a verified retained incident check on later native evidence with the frozen parameters. '
                 'No correction or write occurs; a passing numeric regression does not establish physical repair.'}
    if verification['status'] != 'pass':
        return result
    manifest, _ = _manifest(root)
    check, parameters = manifest['check'], deepcopy(manifest['parameters'])
    source = _native_source(candidate, check)
    if check == 'native_run_context':
        from .native_evidence import fingerprint_native_context

        before = fingerprint_native_context(source)
    else:
        before = _inventory(source) if source.is_dir() else _hash(source)
    report = _retained_run(check, source, parameters, root / 'after' if check in ('native_run_context', 'evaluation_cohort') else None)
    if check == 'trajectory_sampling' and set(report['joint_names']) != set(verification['checks']['before']['joint_names']):
        report = dict(report, status='unknown')
    after = fingerprint_native_context(source) if check == 'native_run_context' else _inventory(source) if source.is_dir() else _hash(source)
    if before != after or _inventory(root, case=True) != case_inventory:
        raise InputError('Incident case or native candidate changed during regression checking')
    # The expected outcome is the retained corrected outcome (pass, or warning for a window audit with advisories).
    expected = verification.get('checks', {}).get('after', {}).get('status', 'pass')
    result.update(status=report['status'], case_verified=True, candidate_evaluated=True, expected_status=expected,
                  candidate=_summary(report), parameters=parameters, check=check)
    return _response_bound(result)
