"""Qualified Arena numeric conversion readset; no converter/model execution."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .artifacts import InputError
from .conversion_profile_data import (
    AUXILIARY_FIELDS,
    CODE_HASHES,
    EXPECTED_CONFIG,
    PATH_FIELDS,
    REFERENCE_HASHES,
    SCALAR_FIELDS,
)
from .native_intake import _directory_fd, _open_file

MAX_ORIGINAL = 12 * 1024 ** 3
MAX_SMALL = 512 * 1024
MAX_REPORT = 2 * 1024 ** 2


class _Prepared(dict):
    """Internal authorized snapshot; never an MCP argument or artifact instruction."""


class _ReferenceBinding(dict):
    """Internal result of bounded reference intake, not a public argument schema."""


_SELECTORS = ('original_recording', 'converter_source', 'converter_config', 'converter_directory')
_SCOPE = 'state_action_and_teleop_numeric_conversion'
_SCALAR_POLICY = {'instruction_max_UTF8_bytes': 4096, 'task_index_min': 0,
                  'task_index_max': 2**63-1, 'integer_fps_min': 1, 'integer_fps_max': 1000}


def _expected_digest(value):
    if value is not None and (type(value) is not str or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise InputError('Expected conversion criterion must be64 lowercase hexadecimal characters')
    return value


def _criterion_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _reference_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise InputError('Conversion reference contains duplicate JSON keys')
            value[key] = item
        return value
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(InputError('Reference requires finite JSON')))
        pending, nodes = [(value, 0)], 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            if nodes > 100000 or depth > 32:
                raise InputError('Conversion reference exceeds depth/collection bounds')
            if type(item) is dict:
                pending.extend((v, depth + 1) for v in item.values())
            elif type(item) is list:
                pending.extend((v, depth + 1) for v in item)
            elif type(item) is float and not math.isfinite(item):
                raise InputError('Conversion reference requires finite JSON')
        return value
    except InputError:
        raise
    except (ValueError, UnicodeError, RecursionError) as error:
        raise InputError('Conversion reference must be supported bounded JSON data') from error


def resolve_conversion_reference(reference=None, expected=None, *, original_recording=None,
        converter_source=None, converter_config=None, converter_directory=None, read_roots=None):
    """Restore only generated conversion selectors under current read authority.

    A saved report is data, not authority, code, authenticated history or a copy
    of the original. Caller/network/output/runtime settings are never restored.
    """
    expected = _expected_digest(expected)
    selectors = dict(zip(_SELECTORS, (original_recording, converter_source, converter_config,
                                     converter_directory), strict=True))
    if reference is None:
        return {'selectors': selectors, 'expected_criterion': expected, 'reference_binding': None}
    if any(value is not None for value in selectors.values()):
        raise InputError('Conversion reference supplies its selectors; omit explicit converter/original selectors')
    roots = _roots(read_roots)
    reference_path = _path(reference, roots)
    raw, artifact = _read_small(reference_path, 8 * 1024 ** 2)
    report = _reference_json(raw)
    return _resolve_reference_document(report, artifact, roots, expected)


def _resolve_reference_document(report, artifact, roots, expected):
    if (type(report) is not dict or type(report.get('schema_version')) is not int
            or report['schema_version'] != 1 or report.get('kind') not in ('dataset_preflight', 'conversion_preflight_reference')
            or type(report.get('components')) is not dict):
        raise InputError('Require a full saved preflight report or generated conversion reference view, not a concise summary')
    if report['kind'] == 'conversion_preflight_reference' and (
            set(report) != {'schema_version', 'kind', 'status', 'reference_available', 'reference_status',
                           'source_report_status', 'source_conversion_status', 'source_error_summary', 'components', 'scope'}
            or report['reference_available'] is not True
            or report['reference_status'] != 'pass'
            or type(report['source_report_status']) is not str
            or report['source_report_status'] not in {'pass', 'error', 'warning', 'unknown'}
            or report['status'] != report['source_report_status']
            or set(report['components']) != {'conversion_fidelity'}):
        raise InputError('Require a complete supported conversion reference view')
    component = report['components'].get('conversion_fidelity')
    if (type(component) is not dict or component.get('kind') != 'conversion_fidelity'
            or type(component.get('schema_version')) is not int or component['schema_version'] != 1
            or component.get('requested_scope') != _SCOPE
            or type(component.get('status')) is not str
            or component['status'] not in {'pass', 'error', 'warning', 'unknown'}):
        raise InputError('Saved report lacks a complete supported conversion component')
    if report['kind'] == 'conversion_preflight_reference' and report['source_conversion_status'] != component['status']:
        raise InputError('Generated reference view has inconsistent source conversion status')
    if report['kind'] == 'conversion_preflight_reference':
        summary = report['source_error_summary']
        if (type(report['scope']) is not str or len(report['scope']) > 1024
                or summary is not None and (type(summary) is not dict or set(summary) != {'code', 'message', 'remedy'}
                    or any(type(value) is not str or len(value) > 160 for value in summary.values()))):
            raise InputError('Generated reference view has malformed bounded scope/error preview')
    reuse, criterion = component.get('conversion_reuse'), component.get('criterion')
    if (type(reuse) is not dict or set(reuse) != {'schema', 'criterion_schema', 'criterion_sha256',
            'criterion_basis_complete', 'inputs', 'requested_scope'}
            or reuse['schema'] != 'conversion_preflight_reuse_v1'
            or reuse['criterion_basis_complete'] is not True or reuse['requested_scope'] != _SCOPE
            or type(criterion) is not dict or reuse['criterion_schema'] != 'arena43_with_teleop_numeric_v2'
            or criterion.get('schema') != reuse['criterion_schema']):
        raise InputError('Reference needs complete generated source-criterion context from this feature; old/incomplete reports cannot be upgraded automatically')
    digest = _expected_digest(component.get('criterion_sha256'))
    if digest is None or reuse['criterion_sha256'] != digest or _criterion_digest(criterion) != digest:
        raise InputError('Saved conversion criterion descriptor/digest does not match')
    if expected is not None and expected != digest:
        raise InputError('Expected conversion criterion conflicts with the saved reference criterion')
    criterion_keys = {'schema', 'original_sha256', 'original_size', 'actual_converter_inputs', 'actual_config',
        'resolved_references', 'source_selection', 'selection_rule', 'required_auxiliary_fields', 'scalar_policy',
        'numeric_semantic_profile', 'semantic_template_hash_scope', 'evaluator', 'dependencies', 'python'}
    evaluator_keys = {'conversion_numeric.py', 'conversion_fidelity.py', 'conversion_fidelity_worker.py',
        'conversion_profile_data.py', 'native_intake.py', 'native_parquet_guard.py', 'artifacts.py',
        'preflight.py', 'cli.py', 'server.py', '__init__.py'}
    profile_keys = set(CODE_HASHES) | {'isaaclab_arena_gr00t/lerobot/config/g1_static_apple_config.yaml'} | {
        EXPECTED_CONFIG[key] for key in ['modality_template_path', 'policy_joints_config_path',
                                        'action_joints_config_path', 'state_joints_config_path']}
    if (set(criterion) != criterion_keys or _expected_digest(criterion.get('original_sha256')) is None
            or type(criterion.get('original_size')) is not int or not 0 < criterion['original_size'] <= MAX_ORIGINAL
            or criterion.get('selection_rule') != 'all_supported_original_numeric_rows_except_terminal_v2'
            or not _equal_typed(criterion.get('scalar_policy'), _SCALAR_POLICY)
            or type(criterion.get('numeric_semantic_profile')) is not dict
            or set(criterion['numeric_semantic_profile']) != profile_keys
            or any(_expected_digest(value) is None for value in criterion['numeric_semantic_profile'].values())
            or criterion.get('semantic_template_hash_scope') != 'Reviewed profile identity only; not a read actual config file'
            or not _equal_typed(criterion.get('required_auxiliary_fields'), json.loads(json.dumps(AUXILIARY_FIELDS)))
            or type(criterion.get('evaluator')) is not dict or set(criterion['evaluator']) != evaluator_keys
            or any(_expected_digest(value) is None for value in criterion['evaluator'].values())
            or type(criterion.get('dependencies')) is not dict
            or set(criterion['dependencies']) != {'numpy', 'h5py', 'pyarrow', 'PyYAML'}
            or any(type(value) is not str or not 0 < len(value) <= 128 for value in criterion['dependencies'].values())
            or type(criterion.get('python')) is not str or not 0 < len(criterion['python']) <= 32):
        raise InputError('Saved criterion has an unsupported source/selection/verifier descriptor')
    prior_binding = component.get('criterion_reuse')
    if prior_binding is not None and (type(prior_binding) is not dict or prior_binding.get('status') != 'pass'
            or prior_binding.get('requested') is not True or prior_binding.get('matches') is not True
            or prior_binding.get('candidate_evaluated_under_expected_criterion') is not True
            or prior_binding.get('expected_sha256') != digest or prior_binding.get('current_sha256') != digest):
        raise InputError('A refused/unestablished reuse cannot silently establish a replacement baseline; use explicit original selectors')
    values = reuse['inputs']
    if type(values) is not dict or set(values) != set(_SELECTORS):
        raise InputError('Reference requires all generated conversion selectors including explicit cwd context')
    selectors = {}
    for key in _SELECTORS:
        value = values[key]
        if value is None and key == 'converter_directory':
            selectors[key] = None
        elif type(value) is str and Path(value).is_absolute():
            selectors[key] = _path(value, roots)
        else:
            raise InputError('Saved conversion selectors must be absolute literal paths (cwd may be null)')
    config, references = criterion.get('actual_config'), criterion.get('resolved_references')
    if (type(config) is not dict or set(config) != set(EXPECTED_CONFIG)
            or type(references) is not dict or set(references) != set(PATH_FIELDS) - {'hdf5_name'}):
        raise InputError('Saved conversion recipe descriptor is incomplete/unsupported')
    # Authorize every extracted role before any actual converter/source open.
    for field in PATH_FIELDS:
        if type(config.get(field)) is not str or not config[field]:
            raise InputError('Saved conversion recipe requires literal path fields')
        if field == 'hdf5_name':
            continue
        if not Path(config[field]).is_absolute() and selectors['converter_directory'] is None:
            raise InputError('Saved relative converter references require the retained explicit converter_directory')
        observed = _path(config[field], roots, selectors['converter_directory'])
        if observed != references[field]:
            raise InputError('Saved resolved reference disagrees with its explicit interpretation context')
    original = _path(config['hdf5_name'], roots, references['data_root'])
    if original != selectors['original_recording']:
        raise InputError('Saved original selector differs from the saved native config-selected path')
    inputs = criterion.get('actual_converter_inputs')
    if type(inputs) is not list or not 1 <= len(inputs) <= 16:
        raise InputError('Saved conversion source identity list is incomplete/bounded')
    by_path = {}
    for item in inputs:
        if (type(item) is not dict or set(item) != {'path', 'sha256', 'size_bytes'}
                or type(item['size_bytes']) is not int or not 0 <= item['size_bytes'] <= MAX_SMALL):
            raise InputError('Saved conversion source identity is malformed')
        path = _path(item['path'], roots)
        if not Path(item['path']).is_absolute() or path in by_path or _expected_digest(item['sha256']) is None:
            raise InputError('Saved conversion source identities require unique absolute locators')
        by_path[path] = item
    if selectors['converter_config'] not in by_path:
        raise InputError('Saved config selector lacks its actual raw-byte identity')
    for name in CODE_HASHES:
        if _path(str(Path(selectors['converter_source']) / name), roots) not in by_path:
            raise InputError('Saved converter selector lacks its semantic source closure')
    for field in REFERENCE_HASHES:
        if references[field] not in by_path:
            raise InputError('Saved reference selector lacks its actual raw-byte identity')
    selection = criterion.get('source_selection')
    if type(selection) is not list or not 1 <= len(selection) <= 256:
        raise InputError('Saved criterion lacks a bounded complete source selection')
    keys, ids = set(), set()
    for row in selection:
        if (type(row) is not list or len(row) != 3 or type(row[0]) is not str or not row[0]
                or len(row[0]) > 256 or row[0] in {'.', '..'} or '/' in row[0]
                or any(ord(c) < 32 for c in row[0]) or type(row[1]) is not int or not 0 <= row[1] <= 255
                or type(row[2]) is not int or not 1 <= row[2] <= 4095 or row[0] in keys or row[1] in ids):
            raise InputError('Saved native source selection is malformed or ambiguous')
        keys.add(row[0])
        ids.add(row[1])
    binding = _ReferenceBinding(artifact=artifact, selectors=selectors, expected_sha256=digest,
                                criterion=criterion, original_status=component.get('status', 'unknown'))
    return {'selectors': selectors, 'expected_criterion': digest, 'reference_binding': binding}


def conversion_reference_view(full_report, max_bytes=192 * 1024):
    """Return a bounded generated criterion view, never a truncated full report.

    This contains selected baseline intent only. Other preflight diagnostics and
    original/export bytes are omitted; later calls revalidate the actual inputs.
    """
    if type(max_bytes) is not int or not 1024 <= max_bytes <= 192 * 1024:
        raise InputError('Conversion reference-view budget must be a strict integer in[1024,196608]')
    scope = ('Generated conversion reference view; other preflight components/findings and source/export bytes are omitted. '
             'Caller-selected intent, not authenticated history or original-report reproduction.')
    statuses = {'pass', 'error', 'warning', 'unknown'}
    def supported_status(value):
        return value if type(value) is str and value in statuses else None
    source_report_status = supported_status(full_report.get('status')) if type(full_report) is dict else None
    components = full_report.get('components') if type(full_report) is dict else None
    original = components.get('conversion_fidelity') if type(components) is dict else None
    source_conversion_status = supported_status(original.get('status')) if type(original) is dict else None
    source_error_summary = None
    def preview(value, limit, fallback):
        value = value[:limit] if type(value) is str else fallback
        while len(json.dumps(value).encode()) > 160:
            value = value[:-1]
        return value
    candidates = (original.get('findings', []) if type(original) is dict else [])
    candidates = (candidates if type(candidates) is list else []) + (
        full_report.get('findings', []) if type(full_report) is dict and type(full_report.get('findings')) is list else [])
    for finding in candidates:
        if type(finding) is dict and finding.get('status') == 'error':
            source_error_summary = {key: preview(finding.get(key), limit, fallback)
                for key, limit, fallback in [('code', 80, 'source.error'), ('message', 120, 'Source preflight contains an ERROR.'),
                                             ('remedy', 120, 'Inspect the complete original preflight report and cited evidence.')]}
            break
    def unavailable(reason):
        result = {'schema_version': 1, 'kind': 'conversion_preflight_reference',
                'status': 'error' if 'error' in (source_report_status, source_conversion_status) else 'unknown',
                'reference_available': False, 'reference_status': 'unknown',
                'source_report_status': source_report_status, 'source_conversion_status': source_conversion_status,
                'source_error_summary': source_error_summary, 'reason': str(reason)[:160], 'scope': scope,
                'next': 'Run the matching CLI with --json --output FILE for the full report; establish complete source context with explicit selectors. Never truncate or reseal a reference.'}
        if len(json.dumps(result, allow_nan=False).encode()) > max_bytes:
            result.update(reason=str(reason)[:80], scope='Reference unavailable; known source errors are preserved. No numeric or historical assurance.',
                          next='Save the complete CLI JSON report; do not truncate or reseal a reference.')
            if source_error_summary is not None:
                result['source_error_summary'] = dict(source_error_summary, message='',
                    remedy='Inspect the complete original CLI report and cited evidence.')
        return result
    try:
        if type(full_report) is not dict or full_report.get('kind') != 'dataset_preflight':
            return unavailable('Require a complete current preflight result')
        _resolve_reference_document(full_report, None, None, None)
        original = full_report['components']['conversion_fidelity']
        required = ['schema_version', 'kind', 'status', 'requested_scope', 'criterion',
                    'criterion_sha256', 'conversion_reuse']
        component = {key: original[key] for key in required}
        if 'criterion_reuse' in original:
            component['criterion_reuse'] = original['criterion_reuse']
        if source_report_status is None:
            return unavailable('Preflight status is malformed')
        view = {'schema_version': 1, 'kind': 'conversion_preflight_reference', 'status': source_report_status,
                'reference_available': True, 'reference_status': 'pass', 'source_report_status': source_report_status,
                'source_conversion_status': source_conversion_status, 'source_error_summary': source_error_summary,
                'components': {'conversion_fidelity': component}, 'scope': scope}
        raw = json.dumps(view, allow_nan=False).encode()
        if len(raw) > max_bytes:
            return unavailable('Complete criterion descriptor exceeds the requested view byte bound; no partial reference was emitted')
        return json.loads(raw)
    except (InputError, KeyError, TypeError, ValueError, RecursionError) as error:
        return unavailable(error)


def _stamp(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _path(value, roots=None, base=None):
    if type(value) is not str or not value or len(value) > 4096 or any(ord(c) < 32 for c in value):
        raise InputError('Conversion input paths must be bounded literal text')
    try:
        if len(value.encode('utf-8')) > 4096:
            raise InputError('Conversion input path exceeds its UTF8 byte bound')
    except UnicodeError as error:
        raise InputError('Conversion input paths require valid UTF8 text') from error
    candidate = Path(value)
    if value.startswith('~') or '..' in candidate.parts or '://' in value:
        raise InputError('Conversion paths require intended literal local paths without tilde/parent components')
    if not candidate.is_absolute():
        candidate = Path(base or Path.cwd()) / candidate
    candidate = Path(os.path.abspath(candidate))
    if roots is not None and not any(candidate == Path(root) or candidate.is_relative_to(Path(root)) for root in roots):
        raise InputError('Conversion input is outside authorized read roots')
    return str(candidate)


def _roots(values):
    if values is None:
        return None
    if type(values) not in (list, tuple) or not values:
        raise InputError('Conversion read roots must be an explicit nonempty path list')
    return [_path(value) for value in values]


def _regular(path):
    leaf = Path(path)
    return _open_file(leaf.parent, leaf.name)


def _read_small(path, limit=MAX_SMALL):
    try:
        fd, before = _regular(path)
    except FileNotFoundError as error:
        raise InputError(f'Conversion input does not exist: {str(path)!r}. Check the input path and rerun.') from error
    except OSError as error:
        raise InputError(f'Conversion input must be an accessible regular file without symlink ancestors: '
                         f'{str(path)!r}. Check the input path and rerun.') from error
    try:
        if before.st_size > limit:
            raise InputError('Conversion metadata/source file exceeds its byte bound')
        chunks, total = [], 0
        while raw := os.read(fd, min(65536, limit + 1 - total)):
            chunks.append(raw)
            total += len(raw)
            if total > limit:
                raise InputError('Conversion metadata/source file grew beyond its byte bound')
        if _stamp(os.fstat(fd)) != _stamp(before) or total != before.st_size:
            raise InputError('Conversion metadata/source file changed while reading')
        raw = b''.join(chunks)
        return raw, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                     'size_bytes': total, 'stamp': _stamp(before)}
    finally:
        os.close(fd)


def _yaml(raw):
    try:
        import yaml
        # Aliases/tags/duplicate keys are unnecessary for this qualified flat profile.
        events = list(yaml.parse(raw))
        depth = 0
        for event in events:
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
                if depth > 32:
                    raise InputError('Conversion YAML exceeds depth32')
            elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
        if len(events) > 4096 or any(isinstance(e, yaml.AliasEvent) or getattr(e, 'tag', None) for e in events):
            raise InputError('Conversion YAML aliases, tags or excessive structure are unsupported')
        class Loader(yaml.SafeLoader):
            pass
        def mapping(loader, node):
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node)
                if type(key) is not str or key in result:
                    raise InputError('Conversion YAML requires unique text keys')
                result[key] = loader.construct_object(value_node)
            return result
        Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        value = yaml.load(raw, Loader=Loader)
        if type(value) is not dict:
            raise InputError('Conversion YAML must be a mapping')
        return value
    except ImportError as error:
        raise InputError('Conversion config parsing requires the optional PyYAML dependency') from error
    except InputError:
        raise
    except Exception as error:
        raise InputError('Conversion YAML is not supported bounded configuration data') from error


def _equal_typed(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(_equal_typed(a, b) for a, b in zip(left, right, strict=False))
    if type(left) is dict:
        return set(left) == set(right) and all(_equal_typed(left[key], right[key]) for key in left)
    return left == right


def _unknown(reason, artifacts=None):
    return {'schema_version': 1, 'kind': 'conversion_fidelity', 'status': 'unknown',
            'requested_scope': 'state_action_and_teleop_numeric_conversion',
            'numeric_association_status': 'unknown', 'source_membership': {'status': 'unknown'},
            'native_consumption': {'status': 'unknown', 'requested': False, 'reason': 'Native consumer was not run'},
            'pixel_fidelity': {'status': 'unknown'}, 'criterion_sha256': None,
            'findings': [{'code': 'conversion.unassessed', 'status': 'unknown', 'message': str(reason)[:1500],
                          'remedy': 'Supply the supported actual converter/config and authorized original/export inputs; no converter is executed.'}],
            'artifacts': artifacts or [], 'coverage': {'whole_numeric_source_to_export': False,
                                                      'whole_dataset_verified': False},
            'scope': 'Conditional source-to-export numeric comparison only; no model, robot, pixel or historical producer proof.'}


def prepare_conversion_inputs(converted_directory, *, original_recording, converter_source,
        converter_config, converter_directory=None, read_roots=None, max_rows=200000, max_files=1024,
        expected_criterion=None, reference_binding=None):
    """Authorize every explicit/extracted path before any unrelated intake/network.

    No HDF5 content, Parquet data or converter code is opened/executed here.
    Actual YAML parsing is data-only; native config constructors are never called.
    """
    expected_criterion = _expected_digest(expected_criterion)
    if reference_binding is not None and type(reference_binding) is not _ReferenceBinding:
        raise InputError('Reference binding must come from the internal saved-report resolver')
    if type(max_rows) is not int or not 1 <= max_rows <= 200000:
        raise InputError('Conversion max_rows must be a strict integer in [1,200000]')
    if type(max_files) is not int or not 1 <= max_files <= 1024:
        raise InputError('Conversion max_files must be a strict integer in [1,1024]')
    roots = _roots(read_roots)
    paths = {key: _path(value, roots) for key, value in [
        ('converted_directory', converted_directory), ('original_recording', original_recording),
        ('converter_source', converter_source), ('converter_config', converter_config)]}
    if converter_directory is not None:
        converter_directory = _path(converter_directory, roots)
    if reference_binding is not None:
        selectors = {key: paths[key] for key in _SELECTORS if key != 'converter_directory'}
        selectors['converter_directory'] = converter_directory
        if selectors != reference_binding['selectors'] or expected_criterion != reference_binding['expected_sha256']:
            raise InputError('Restored reference context does not match these authorized inputs/criterion')
        _, current = _read_small(reference_binding['artifact']['path'], 8 * 1024 ** 2)
        if current != reference_binding['artifact']:
            raise InputError('Saved conversion reference changed after bounded intake')
    # All explicit paths have been lexically authorized before the first open.
    raw, config_artifact = _read_small(paths['converter_config'], 128 * 1024)
    config = _yaml(raw)
    for field in PATH_FIELDS:
        if type(config.get(field)) is not str or not config[field]:
            raise InputError('Conversion config requires explicit literal ' + field)
    references = {}
    for field in PATH_FIELDS:
        if field == 'hdf5_name':
            continue
        if not Path(config[field]).is_absolute() and converter_directory is None:
            raise InputError('Relative native converter config paths require explicit converter_directory; config/source directories do not imply cwd')
        references[field] = _path(config[field], roots, converter_directory)
    # Native Path(data_root)/hdf5_name selection, without tilde/parent rewriting.
    original_selected = _path(config['hdf5_name'], roots, references['data_root'])
    if original_selected != paths['original_recording']:
        raise InputError('Actual original_recording must match the file selected by actual data_root/hdf5_name')
    code_paths = {name: _path(str(Path(paths['converter_source']) / name), roots) for name in CODE_HASHES}
    artifacts = [config_artifact]
    reasons = []
    if set(config) != set(EXPECTED_CONFIG) or any(
            key not in (*PATH_FIELDS, *SCALAR_FIELDS) and not _equal_typed(value, EXPECTED_CONFIG.get(key))
            for key, value in config.items()):
        reasons.append('Actual YAML changes unsupported semantic fields; only paths and bounded instruction/task_index/fps may differ')
    instruction = config.get('language_instruction')
    try:
        valid_instruction = type(instruction) is str and bool(instruction.strip()) and len(instruction.encode('utf-8')) <= 4096
    except UnicodeEncodeError:
        valid_instruction = False
    if not valid_instruction:
        reasons.append('Actual instruction must be nonempty UTF8 text of at most4096bytes')
    if type(config.get('task_index')) is not int or not 0 <= config['task_index'] <= 2 ** 63 - 1:
        reasons.append('Actual task_index must be a strict integer in [0,2**63-1]')
    if type(config.get('fps')) is not int or not 1 <= config['fps'] <= 1000:
        reasons.append('Actual fps must be a strict integer in [1,1000]; this validates a numeric declaration, not physical/camera timing')
    # Even unsupported semantics cannot hide an unauthorized extracted reference.
    plan = _Prepared(paths=paths, converter_directory=converter_directory, read_roots=roots,
                     max_rows=max_rows, max_files=max_files, config=config, references=references,
                     code_paths=code_paths, artifacts=artifacts, supported=False, reasons=reasons,
                     expected_criterion=expected_criterion, reference_binding=reference_binding)
    if reasons:
        return plan
    unique_paths = {paths['converter_config'], *code_paths.values(), *[references[k] for k in REFERENCE_HASHES]}
    if len(unique_paths) + 4 > max_files:
        plan['reasons'].append('Conversion file bound cannot cover the fixed input/metadata readset')
        return plan
    for directory in [paths['converted_directory'], paths['converter_source'], references['data_root'],
                      *([converter_directory] if converter_directory else [])]:
        try:
            fd = _directory_fd(directory)
        except OSError as error:
            raise InputError('Conversion directories must be accessible without symlink ancestors') from error
        os.close(fd)
    seen = {config_artifact['path']: config_artifact}
    for path, expected in [*[(code_paths[n], h) for n, h in CODE_HASHES.items()],
                            *[(references[n], h) for n, h in REFERENCE_HASHES.items()]]:
        if path not in seen:
            _, artifact = _read_small(path)
            seen[path] = artifact
        if seen[path]['sha256'] != expected:
            reasons.append('Unsupported or drifted converter/reference bytes at ' + path)
    plan['artifacts'] = list(seen.values())
    plan['supported'] = not reasons
    return plan


def _check_conversion_fidelity(converted_directory, *, original_recording, converter_source,
        converter_config, converter_directory=None, read_roots=None, max_rows=200000, max_files=1024,
        expected_criterion=None, reference_binding=None, _prepared=None):
    if _prepared is None:
        plan = prepare_conversion_inputs(converted_directory, original_recording=original_recording,
            converter_source=converter_source, converter_config=converter_config,
            converter_directory=converter_directory, read_roots=read_roots, max_rows=max_rows, max_files=max_files,
            expected_criterion=expected_criterion, reference_binding=reference_binding)
    else:
        if type(_prepared) is not _Prepared:
            raise InputError('Prepared conversion inputs must come from the internal authorizer')
        plan = _prepared
        roots = _roots(read_roots)
        expected = {k: _path(v, roots) for k, v in [('converted_directory', converted_directory),
            ('original_recording', original_recording), ('converter_source', converter_source),
            ('converter_config', converter_config)]}
        directory = _path(converter_directory, roots) if converter_directory is not None else None
        if (expected != plan['paths'] or roots != plan['read_roots'] or directory != plan['converter_directory']
                or type(max_rows) is not int or type(max_files) is not int
                or max_rows != plan['max_rows'] or max_files != plan['max_files']
                or _expected_digest(expected_criterion) != plan.get('expected_criterion')
                or reference_binding is not plan.get('reference_binding')):
            raise InputError('Prepared conversion plan does not match these explicit inputs/bounds')
    if not plan['supported']:
        return _unknown('; '.join(plan['reasons']), plan['artifacts'])
    if os.name != 'posix' or any(importlib.util.find_spec(name) is None for name in ['h5py', 'pyarrow']):
        return _unknown('Conversion numeric reader requires POSIX and optional dataset dependencies', plan['artifacts'])
    code = ('import resource,sys; '
            'resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296)); '
            'resource.setrlimit(resource.RLIMIT_CPU,(300,300)); '
            'resource.setrlimit(resource.RLIMIT_FSIZE,(67108864,67108864)); '
            'resource.setrlimit(resource.RLIMIT_NOFILE,(128,128)); '
            'sys.path.insert(0,sys.argv[1]); '
            'from policy_doctor.conversion_fidelity_worker import main; main(sys.argv[2]); '
            # The report is written and closed by now. Skip interpreter teardown:
            # after reading Parquet through Python file objects, pyarrow 17 can
            # abort at exit, turning a complete report into a nonzero refusal.
            'sys.stdout.flush(); sys.stderr.flush(); import os; os._exit(0)')
    env = dict(os.environ, PYTHONPATH='', HDF5_PLUGIN_PATH='', HDF5_PLUGIN_PRELOAD='::',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    with tempfile.TemporaryDirectory(prefix='conversion-numeric-worker-') as temporary:
        output = Path(temporary) / 'result.json'
        request = json.dumps(dict(plan), allow_nan=False).encode()
        if len(request) > 512 * 1024:
            return _unknown('Authorized conversion request exceeds worker bound', plan['artifacts'])
        try:
            process = subprocess.run([sys.executable, '-I', '-B', '-c', code,
                str(Path(__file__).absolute().parent.parent), str(output)], input=request,
                cwd=temporary, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=600, check=False)
            if process.returncode or not output.is_file() or output.stat().st_size > MAX_REPORT:
                return _unknown('Bounded numeric worker refused/unavailable; no complete conversion verdict', plan['artifacts'])
            result = json.loads(output.read_bytes())
            if type(result) is not dict or result.get('kind') != 'conversion_fidelity':
                return _unknown('Bounded numeric worker returned an unsupported report', plan['artifacts'])
            return result
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            return _unknown('Bounded numeric worker unavailable: ' + type(error).__name__, plan['artifacts'])


def _field_path(value):
    if len(value) <= 256:
        return value
    return value[:160] + '#sha256:' + hashlib.sha256(value.encode()).hexdigest()


def _drift_fields(before, after):
    """Bounded field categories; avoid echoing arbitrary report/config values."""
    changed, pending = [], [('', before, after)]
    while pending:
        path, left, right = pending.pop()
        if type(left) is dict and type(right) is dict:
            for key in sorted(set(left) | set(right), reverse=True):
                pending.append((path + '/' + str(key), left.get(key), right.get(key)))
        elif not _equal_typed(left, right):
            changed.append(_field_path(path or '/'))
    return changed[:16], max(0, len(changed) - 16)


def _apply_conversion_binding(result, expected, reference):
    if expected is None:
        return result
    current = result.get('criterion_sha256')
    complete = result.get('conversion_reuse', {}).get('criterion_basis_complete') is True
    matches = current == expected if current is not None and complete else None
    fields, omitted = [], 0
    if reference is not None and type(result.get('criterion')) is dict:
        fields, omitted = _drift_fields(reference['criterion'], result['criterion'])
        identity_keys = ['original_sha256', 'original_size', 'actual_converter_inputs', 'actual_config',
                         'resolved_references', 'evaluator', 'dependencies', 'python', 'numeric_semantic_profile',
                         'scalar_policy', 'selection_rule', 'required_auxiliary_fields']
        if matches is None and any(not _equal_typed(reference['criterion'][key], result['criterion'][key])
                                   for key in identity_keys):
            # An independently verified frozen-input/verifier difference is
            # still known when another source subset could not be modeled.
            matches = False
    # Even an unsupported new recipe may prove a changed raw identity. This is
    # a verified frozen-input difference, not a numerical diagnosis of that recipe.
    if reference is not None and current is None:
        saved = {item['path']: item for item in reference['criterion']['actual_converter_inputs']}
        for artifact in result.get('artifacts', []):
            if artifact.get('path') in saved and any(artifact.get(k) != saved[artifact['path']][k]
                    for k in ['sha256', 'size_bytes']):
                fields.append(_field_path('/actual_converter_inputs/' + artifact['path']))
        if fields:
            matches = False
    status = 'pass' if matches is True else 'error' if matches is False else 'unknown'
    binding = {'requested': True, 'status': status, 'expected_sha256': expected,
               'current_sha256': current, 'matches': matches,
               'candidate_evaluated_under_expected_criterion': matches is True,
               'reference_artifact': dict(reference['artifact']) if reference is not None else None,
               'reference_original_status': reference['original_status'] if reference is not None else None,
               'reference_scope': 'Caller-selected baseline intent; not authenticated history or reproduced original report',
               'original_report_reproduced': False,
               'drift_fields': fields[:16], 'drift_fields_omitted': omitted + max(0, len(fields) - 16)}
    result['criterion_reuse'] = binding
    if matches is True:
        return result
    keys = ['status', 'numeric_association_status', 'joint_numeric_association_status',
            'auxiliary_numeric_association_status', 'source_membership', 'coverage']
    result['current_criterion_observations'] = {key: result[key] for key in keys if key in result}
    for key in ['numeric_association_status', 'joint_numeric_association_status', 'auxiliary_numeric_association_status']:
        if key in result:
            result[key] = 'unknown'
    result['source_membership'] = {'status': 'unknown', 'reason': 'Not established under the requested fixed criterion; see current_criterion_observations'}
    result['coverage'] = dict(result.get('coverage', {}), whole_numeric_source_to_export=False)
    for field in result.get('auxiliary_fields', {}).values():
        field['observed_under_current_criterion_status'] = field.get('status', 'unknown')
        field['status'] = 'unknown'
    for finding in result.get('findings', []):
        finding['observed_criterion_sha256'] = current
    counts = dict(result.get('counts', {'error': sum(f.get('status') == 'error' for f in result.get('findings', [])),
                                      'unknown': sum(f.get('status') == 'unknown' for f in result.get('findings', []))}))
    counts[status] = counts.get(status, 0) + 1
    finding = {'status': status, 'code': 'conversion.criterion_drift' if status == 'error' else 'conversion.criterion_unestablished',
               'message': ('The fixed conversion criterion changed; this candidate was not accepted under the saved criterion.' if status == 'error'
                           else 'The fixed conversion criterion could not be completely revalidated; no expected-criterion numeric verdict is established.'),
               'remedy': 'Restore the intended original/recipe/verifier, or explicitly establish a new baseline with original selectors; never replace the expected digest automatically.'}
    findings = [finding, *result.get('findings', [])]
    result.update(status='error' if counts.get('error', 0) else 'unknown', counts=counts,
                  findings=findings[:64], findings_total=sum(counts.values()),
                  findings_truncated=sum(counts.values()) > min(len(findings), 64))
    return result


def check_conversion_fidelity(converted_directory, *, original_recording, converter_source,
        converter_config, converter_directory=None, read_roots=None, max_rows=200000, max_files=1024,
        expected_criterion=None, reference_binding=None, _prepared=None):
    """Run the same numeric check and enforce an optional fixed-criterion assertion."""
    expected_criterion = _expected_digest(expected_criterion)
    if reference_binding is not None and type(reference_binding) is not _ReferenceBinding:
        raise InputError('Reference binding must come from the internal saved-report resolver')
    result = _check_conversion_fidelity(converted_directory, original_recording=original_recording,
        converter_source=converter_source, converter_config=converter_config,
        converter_directory=converter_directory, read_roots=read_roots, max_rows=max_rows, max_files=max_files,
        expected_criterion=expected_criterion, reference_binding=reference_binding, _prepared=_prepared)
    if reference_binding is not None:
        _, current = _read_small(reference_binding['artifact']['path'], 8 * 1024 ** 2)
        if current != reference_binding['artifact']:
            raise InputError('Saved conversion reference changed during candidate checking')
    return _apply_conversion_binding(result, expected_criterion, reference_binding)
