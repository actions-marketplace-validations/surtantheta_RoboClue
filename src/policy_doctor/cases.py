"""Portable native trajectory checks; declarative inputs, never embedded code."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import __version__
from .artifacts import MAX_JSON_BYTES, InputError, read_json
from .publication import publish_capture
from .trajectory import compare_joint_trajectories, sample_joint_trajectory

KIND = 'trajectory_regression_case'
SAMPLE_KIND = 'native_trajectory_sample_case'
SCOPE = ('Reproduces recorded sampled-reference findings, including differences and unknowns. '
         'A reproduced finding does not establish a correct diagnosis, physical repair or robot execution. '
         'File hashes check consistency with this case, not author authenticity.')
PARAMETERS = {'sample_times', 'reference_method', 'candidate_method', 'control_method', 'position_limits'}
SAMPLE_PARAMETERS = {'sample_times', 'interpolation_method', 'position_limits'}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _encode(value: dict) -> bytes:
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')) + '\n').encode()
    if len(raw) > MAX_JSON_BYTES:
        raise InputError('Case report exceeds the 8 MiB JSON budget; use fewer sample times')
    return raw


def _bytes(path: Path) -> bytes:
    if not path.is_file():
        raise InputError(f'Expected a regular case input: {path.name}')
    with path.open('rb') as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise InputError('Case input exceeds the 8 MiB JSON budget')
    return raw


def _fingerprint(yaml_used=False) -> dict:
    root = Path(__file__).parent
    result = {'engine_version': __version__, 'numpy_version': np.__version__,
            'module_sha256': {name: _digest((root / name).read_bytes())
                              for name in ('cases.py', 'trajectory.py', 'artifacts.py')}}
    if yaml_used:
        result['pyyaml_version'] = importlib.metadata.version('PyYAML')
    return result


def _portable(value, paths):
    if isinstance(value, list):
        return [_portable(v, paths) for v in value]
    if isinstance(value, dict):
        result = {key: _portable(v, paths) for key, v in value.items()}
        # Normalize only artifact locators; native header text remains literal.
        if 'sha256' in value and isinstance(value.get('path'), str) and value['path'] in paths:
            result['path'] = paths[value['path']]
        return result
    return value


def _input_names(files, has_control):
    roles = ['reference', 'candidate'] + (['control'] if has_control else [])
    names = {}
    for role in roles:
        matches = [role + suffix for suffix in ('.json', '.yaml', '.yml') if role + suffix in files]
        if len(matches) != 1:
            raise InputError('Require exactly one native JSON/YAML file per case role')
        names[role] = matches[0]
    if set(files) - set(names.values()) - {'expected.json'}:
        raise InputError('Invalid fixed native case file set')
    return names


def _evaluate(root: Path, parameters: dict, has_control: bool) -> dict:
    present = {role + suffix for role in ('reference', 'candidate', 'control')
               for suffix in ('.json', '.yaml', '.yml') if (root / (role + suffix)).exists()}
    names = _input_names(present, has_control)
    paths = {str((root / name).resolve()): name for name in names.values()}
    options = {'reference_method': parameters['reference_method'], 'position_limits': parameters['position_limits']}
    checks = {'candidate': compare_joint_trajectories(str(root / names['reference']), str(root / names['candidate']),
              parameters['sample_times'], candidate_method=parameters['candidate_method'], **options)}
    if has_control:
        checks['control'] = compare_joint_trajectories(str(root / names['reference']), str(root / names['control']),
            parameters['sample_times'], candidate_method=parameters['control_method'], **options)
    return _portable(checks, paths)


def export_trajectory_case(reference: str, candidate: str, destination: str, sample_times: list[float], *,
        control: str | None = None, reference_method: str = 'splines', candidate_method: str = 'splines',
        control_method: str = 'splines', position_limits: dict | None = None) -> dict:
    """Snapshot native JSON/YAML plus exact local comparisons into an exclusively new directory."""
    target = Path(destination).expanduser().absolute()
    target = target.parent.resolve() / target.name
    if target.exists() or target.is_symlink():
        raise InputError('Case destination already exists; choose a new directory')
    parameters = deepcopy({'sample_times': sample_times, 'reference_method': reference_method,
                  'candidate_method': candidate_method, 'control_method': control_method, 'position_limits': position_limits})
    # No author-only absolute paths enter the distributed case.
    inputs, names = {}, {}
    for role, source in [('reference', reference), ('candidate', candidate)] + ([('control', control)] if control is not None else []):
        path = Path(source).expanduser()
        suffix = path.suffix.lower() if path.suffix.lower() in ('.yaml', '.yml') else '.json'
        names[role] = role + suffix
        inputs[names[role]] = _bytes(path)
    yaml_used = any(name.endswith(('.yaml', '.yml')) for name in names.values())
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.policy-doctor-case-', dir=target.parent))
    try:
        for name, raw in inputs.items():
            (staging / name).write_bytes(raw)
        observed = _evaluate(staging, parameters, control is not None)
        expected = _encode(observed)
        (staging / 'expected.json').write_bytes(expected)
        files = {name: _digest(raw) for name, raw in inputs.items()}
        files['expected.json'] = _digest(expected)
        manifest = {'schema_version': 1, 'kind': KIND, 'verifier': _fingerprint(yaml_used), 'parameters': parameters,
                    'has_control': control is not None, 'files': files}
        encoded = _encode(manifest)
        (staging / 'manifest.json').write_bytes(encoded)
        (staging / 'README.md').write_text(
            '# Native trajectory regression case\n\n'
            'Install the recorded policy-deployment-doctor version and NumPy version; YAML inputs also need the ros extra and recorded PyYAML version. Then run:\n\n'
            f"Supplied wheel installation (replace /path/to with its actual location):\n\n```sh\npython -m pip install '/path/to/policy_deployment_doctor-{__version__}-py3-none-any.whl{'[ros]' if yaml_used else ''}' 'numpy=={np.__version__}'"
            + (f" 'PyYAML=={manifest['verifier']['pyyaml_version']}'" if yaml_used else '') + '\n```\n\n'
            '```sh\npolicy-doctor reproduce-case .\n```\n\n'
            'Or call the read-only MCP tool reproduce_trajectory_case with this directory.\n\n'
            'With the mcp extra installed, launch stdio using `policy-doctor serve --root /absolute/path/to/this-case`; the read-root option is `--root`.\n\n'
            'manifest.json records versions, source fingerprints, native input hashes and sample parameters. '
            'expected.json contains the recorded findings with relative evidence locators. '
            'A reproduced difference or unknown is a successful reproduction, not an equal trajectory or physical repair.\n',
            encoding='utf-8')
        (staging / 'READY.json').write_bytes(_encode({'schema_version': 1, 'manifest_sha256': _digest(encoded)}))
        publish_capture(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {'schema_version': 1, 'kind': 'exported_trajectory_case', 'status': 'pass', 'directory': str(target),
            'checks': {name: report['status'] for name, report in observed.items()},
            'files': files, 'verifier': manifest['verifier'], 'scope': SCOPE}


def _file(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise InputError('Case files must be local regular files, not symlinks')
    return path


def _sample_name(files):
    matches = [name for name in ('trajectory.json', 'trajectory.yaml', 'trajectory.yml') if name in files]
    if len(matches) != 1 or set(files) - {matches[0], 'expected.json'}:
        raise InputError('Require exactly one fixed native trajectory file in a sample case')
    return matches[0]


def _evaluate_sample(root: Path, parameters: dict) -> dict:
    present = {name for name in ('trajectory.json', 'trajectory.yaml', 'trajectory.yml') if (root / name).exists()}
    name = _sample_name(present)
    report = sample_joint_trajectory(str(root / name), parameters['sample_times'],
        interpolation_method=parameters['interpolation_method'], position_limits=parameters['position_limits'])
    return _portable({'sample': report}, {str((root / name).resolve()): name})


def _sample_summary(report):
    result = {key: report[key] for key in ('status', 'sampling_complete', 'interpolation_method',
        'execution_status', 'physical_cause', 'position_limits', 'artifacts', 'excursion_count')}
    result.update(sample_count=len(report['samples']), samples=report['samples'][:8],
        samples_truncated=len(report['samples']) > 8,
        waypoint_envelope_excursions=report['waypoint_envelope_excursions'][:8],
        sampled_limit_violations=report['sampled_limit_violations'][:8],
        sampled_limit_violation_count=len(report['sampled_limit_violations']),
        sampled_limit_violations_truncated=len(report['sampled_limit_violations']) > 8)
    return result


def export_sample_case(trajectory: str, destination: str, sample_times: list[float], *,
        interpolation_method: str = 'splines', position_limits: dict | None = None) -> dict:
    """Publish one original native message and its selected static sampling check."""
    target = Path(destination).expanduser().absolute()
    target = target.parent.resolve() / target.name
    if target.exists() or target.is_symlink():
        raise InputError('Case destination already exists; choose a new directory')
    parameters = deepcopy({'sample_times': sample_times, 'interpolation_method': interpolation_method,
                           'position_limits': position_limits})
    source = Path(trajectory).expanduser()
    suffix = source.suffix.lower()
    if suffix not in ('.json', '.yaml', '.yml'):
        raise InputError('Native trajectory extension must be .json, .yaml or .yml')
    name, raw = 'trajectory' + suffix, _bytes(source)
    yaml_used = suffix != '.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.policy-doctor-sample-', dir=target.parent))
    try:
        (staging / name).write_bytes(raw)
        observed = _evaluate_sample(staging, parameters)
        # Refuse a file changed while being evaluated; no hostile-filesystem isolation claim.
        if _bytes(staging / name) != raw:
            raise InputError('Case input changed while it was being evaluated')
        expected = _encode(observed)
        (staging / 'expected.json').write_bytes(expected)
        files = {name: _digest(raw), 'expected.json': _digest(expected)}
        manifest = {'schema_version': 1, 'kind': SAMPLE_KIND, 'verifier': _fingerprint(yaml_used),
                    'parameters': parameters, 'files': files}
        encoded = _encode(manifest)
        (staging / 'manifest.json').write_bytes(encoded)
        (staging / 'README.md').write_text(
            '# Native trajectory sampling case\n\n'
            'This case contains one original native message and the recorded static sampling findings. '
            'Install the recorded policy-deployment-doctor and NumPy versions; YAML also needs the ros extra '
            'and recorded PyYAML version. No original source directory or robot connection is needed.\n\n'
            'Supplied wheel installation (replace /path/to with its actual location):\n\n'
            f"```sh\npython -m pip install '/path/to/policy_deployment_doctor-{__version__}-py3-none-any.whl{'[ros]' if yaml_used else ''}' 'numpy=={np.__version__}'"
            + (f" 'PyYAML=={manifest['verifier']['pyyaml_version']}'" if yaml_used else '') + '\n```\n\n'
            '```sh\npolicy-doctor reproduce-case .\n```\n\n'
            'The existing read-only MCP tool reproduce_trajectory_case runs the same check. '
            'manifest.json records the exact parameters, source fingerprints and original-byte hashes; '
            'expected.json retains the complete recorded findings. Reproduction verifies those findings, '
            'including declared-bound errors and missing-state unknowns. It does not certify a healthy robot, '
            'physical repair or author authenticity.\n', encoding='utf-8')
        (staging / 'READY.json').write_bytes(_encode({'schema_version': 1, 'manifest_sha256': _digest(encoded)}))
        publish_capture(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {'schema_version': 1, 'kind': 'exported_trajectory_sample_case', 'status': 'pass',
        'sample_status': observed['sample']['status'], 'sampling': observed['sample'], 'directory': str(target),
        'files': files, 'verifier': manifest['verifier'], 'scope': SCOPE}


def reproduce_trajectory_case(directory: str) -> dict:
    """Verify a committed case and rerun its allowlisted arithmetic without writes."""
    root = Path(directory).expanduser().resolve()
    ready_path, manifest_path = (_file(root, name) for name in ('READY.json', 'manifest.json'))
    ready_raw, manifest_raw = _bytes(ready_path), _bytes(manifest_path)
    ready, manifest = read_json(ready_path), read_json(manifest_path)
    if (set(ready) != {'schema_version', 'manifest_sha256'} or type(ready['schema_version']) is not int or
            ready['schema_version'] != 1 or ready['manifest_sha256'] != _digest(manifest_raw)):
        raise InputError('Case commit marker or manifest hash is invalid')
    is_sample = manifest.get('kind') == SAMPLE_KIND
    required = {'schema_version', 'kind', 'verifier', 'parameters', 'files'} | (set() if is_sample else {'has_control'})
    if (set(manifest) != required or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or
            manifest['kind'] not in (KIND, SAMPLE_KIND) or (not is_sample and type(manifest['has_control']) is not bool)):
        raise InputError('Unsupported declarative case manifest')
    parameters, files = manifest['parameters'], manifest['files']
    if (not isinstance(parameters, dict) or set(parameters) != (SAMPLE_PARAMETERS if is_sample else PARAMETERS) or
            not isinstance(files, dict) or 'expected.json' not in files):
        raise InputError('Invalid case parameters or fixed native file set')
    names = {'sample': _sample_name(files)} if is_sample else _input_names(files, manifest['has_control'])
    for name, expected_hash in files.items():
        if (not isinstance(expected_hash, str) or len(expected_hash) != 64 or
                any(c not in '0123456789abcdef' for c in expected_hash) or _digest(_bytes(_file(root, name))) != expected_hash):
            raise InputError(f'Case artifact hash mismatch: {name}')
    expected = read_json(_file(root, 'expected.json'))
    observed = _evaluate_sample(root, parameters) if is_sample else _evaluate(root, parameters, manifest['has_control'])
    recorded_verifier, current_verifier = manifest['verifier'], _fingerprint(any(name.endswith(('.yaml', '.yml')) for name in names.values()))
    verifier_matches = recorded_verifier == current_verifier
    reproduction_matches = _encode(expected) == _encode(observed)
    # Refuse changes during checking; no claim of hostile concurrent filesystem isolation.
    if (_bytes(ready_path) != ready_raw or _bytes(manifest_path) != manifest_raw or
            any(_digest(_bytes(_file(root, name))) != value for name, value in files.items())):
        raise InputError('Case changed while it was being checked')
    checks = {} if is_sample else {name: {key: report[key] for key in ('status', 'comparison_complete', 'joint_alignment',
              'same_waypoint_positions_and_times', 'sampled_references_equal', 'first_position_difference',
              'different_sample_fields', 'execution_status', 'physical_cause')} for name, report in observed.items()}
    for name, report in observed.items() if not is_sample else []:
        checks[name]['sampling'] = {role: {key: report[role][key] for key in ('status', 'sampling_complete',
            'position_limits', 'sampled_limit_violations', 'excursion_count', 'artifacts')} for role in ('reference', 'candidate')}
        for sampling in checks[name]['sampling'].values():
            violations = sampling['sampled_limit_violations']
            sampling['sampled_limit_violation_count'] = len(violations)
            sampling['sampled_limit_violations_truncated'] = len(violations) > 8
            sampling['sampled_limit_violations'] = violations[:8]
    if is_sample:
        checks['sample'] = _sample_summary(observed['sample'])
    return {'schema_version': 1, 'kind': 'trajectory_case_reproduction',
            'status': 'unknown' if not verifier_matches else 'pass' if reproduction_matches else 'error',
            'reproduction_matches': reproduction_matches, 'verifier_matches': verifier_matches,
            'recorded_verifier': recorded_verifier, 'current_verifier': current_verifier,
            'version_notice': None if verifier_matches else 'Verifier code, engine version or NumPy differs; inspect results before accepting reproduction.',
            'checked_files': files, 'checks': checks,
            'recorded_findings_sha256': _digest(_encode(expected)), 'current_findings_sha256': _digest(_encode(observed)),
            'scope': SCOPE}
