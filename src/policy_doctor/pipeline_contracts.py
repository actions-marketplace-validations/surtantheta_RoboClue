"""Small, version-bound checks of resolved initialization and camera inputs.

Remote tools only read JSON and source text. Native Python callers can capture
the config/batch they actually use; a declaration is never a runtime attestation.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import CodeType, MethodType

from .artifacts import InputError, contained
from .conversion_fidelity import _read_small, _reference_json
from .fs_guard import open_regular

KINDS = ('openpi_initialization', 'lerobot_camera_inputs')
OPENPI_COMMIT = '3e0325a5e6032a9ab6beef7395fe47ca724980a5'
LEROBOT_COMMIT = '77710dee471d5cfb070fad1adc30c8f94dc14757'
OPENPI_FILE = 'scripts/train_pytorch.py'
OPENPI_HASH = 'baa66f1642e955d5cc2ec3f54d9495035df61d71b028b98ab4ebb1e240548311'
CAMERA_PROFILES = {
    'pi0': ('lerobot/policies/pi0/modeling_pi0.py',
            {'16f086db3cf1a1191a02579f593991d22e4d4cf5fe90c884d78ab6c3010a7a36': (LEROBOT_COMMIT, 1016),
             'd49e0ade33734a0723d82198a4774ddb9745c004d45feb4a8db1aec528b8af77': ('pypi:lerobot==0.6.1', 995)}, '_preprocess_images'),
    'pi05': ('lerobot/policies/pi05/modeling_pi05.py',
             {'ad7315dbbadea5aab50fa7ac7f411aee07d3fbe937a20306eaa1d0beadb70d50': (LEROBOT_COMMIT, 1378),
              '8faec89af9d31f5c2f0e12b4ea369d2af6bf0c25050a13adcaefbd9b8aa03b69': ('pypi:lerobot==0.6.1', 964)}, '_preprocess_images'),
    'smolvla': ('lerobot/policies/smolvla/modeling_smolvla.py',
                {'919cb4223e337067d6728d03d10d6ca730b65adb0cd59969ac25e11f900929ab': (LEROBOT_COMMIT, 391),
                 '37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d': ('pypi:lerobot==0.6.1', 340)}, 'prepare_images'),
}
SCOPE = ('Resolved fields supplied by the caller plus exact audited source files. '
         'The active process, model weights, task performance and physical robot are not attested.')


def _text(value, label, *, limit=256):
    if type(value) is not str or not value or len(value) > limit or not value.isprintable():
        raise InputError(f'{label} must be nonempty printable text of at most {limit} characters')
    return value


def _keys(value, label, *, empty=False):
    if type(value) is not list or len(value) > 8 or (not value and not empty):
        raise InputError(f'{label} must be a list of at most eight camera keys')
    for key in value:
        _text(key, label, limit=160)
        if not key.startswith('observation.images.'):
            raise InputError(f'{label} must use native observation.images.* keys')
    if len(value) != len(set(value)):
        raise InputError(f'{label} must not contain duplicate keys')
    return value


def _source(root, relative, hashes):
    """Read only the named file in a checkout or installed-package layout."""
    if root is None:
        return None, 'Supply runtime_source_root (CLI --runtime-source-root) for the actual consumer.', []
    if not isinstance(root, (str, os.PathLike)):
        raise InputError('runtime_source_root must be a local path')
    _text(os.fspath(root), 'runtime_source_root', limit=4096)
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise InputError('runtime_source_root must be an existing local source directory')
    choices = [relative] if relative.startswith('scripts/') else ['src/' + relative, relative]
    if root.name == 'lerobot' and relative.startswith('lerobot/'):
        choices.append(relative.removeprefix('lerobot/'))
    candidates = [contained(root, name) for name in choices]
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        return None, f'Expected one unambiguous {relative} under the supplied source root; found {len(existing)}.', []
    path = existing[0]
    with open_regular(path, 'pipeline source') as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        return None, f'{relative} exceeds the 2 MiB source bound.', []
    digest = hashlib.sha256(raw).hexdigest()
    evidence = [{'path': path.relative_to(root).as_posix(), 'root': 'runtime_source_root', 'sha256': digest}]
    if digest not in hashes:
        return None, f'{relative} has an unprofiled SHA256 {digest}; its behavior was not assumed.', evidence
    revision, line = hashes[digest]
    evidence[0]['line'] = line
    return {'revision': revision, 'file': evidence[0]['path'], 'line': line,
            'sha256': digest, 'selection': 'exact source file hash; not a whole-checkout attestation'}, '', evidence


def _report(document, status, code, message, remedy, *, evidence=None, profile=None, **details):
    return {'schema_version': 1, 'kind': 'pipeline_contract_check', 'contract_kind': document['kind'],
            'status': status, 'rule_id': 'RC010' if document['kind'] == KINDS[0] else 'RC011',
            'contract_id': 'OP-PT-WEIGHTS-UNSET' if document['kind'] == KINDS[0] else 'LR-CAM-SLOT-SHIFT',
            'scope': SCOPE, 'source_profile': profile, 'native_execution': 'not_run',
            'findings': [{'code': code, 'status': status, 'message': message,
                          'remedy': remedy, 'evidence': evidence or []}], **details}


def capture_openpi_initialization(config, *, backend='pytorch', expected_initialization='pretrained', runtime_source_root=None):
    """Capture the existing trainer's resolved config, not its requested CLI defaults."""
    for name in ('pytorch_weight_path', 'weight_loader', 'resume'):
        if not hasattr(config, name):
            raise InputError('The resolved OpenPI config lacks ' + name)
    loader = config.weight_loader
    path = config.pytorch_weight_path
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    loader_path = getattr(loader, 'params_path', None)
    document = {'schema_version': 1, 'kind': KINDS[0], 'backend': backend,
                'expected_initialization': expected_initialization,
                'train_config': {'name': getattr(config, 'name', 'resolved-config'),
                                 'resume': config.resume, 'pytorch_weight_path': path,
                                 'weight_loader': {'class': type(loader).__module__ + '.' + type(loader).__name__,
                                                   'params_path': str(loader_path) if loader_path is not None else None}}}
    if runtime_source_root is None and type(config).__module__ == 'openpi.training.config':
        origin = inspect.getsourcefile(type(config))
        if origin and Path(origin).as_posix().endswith('/src/openpi/training/config.py'):
            runtime_source_root = Path(origin).resolve().parents[3]
    if runtime_source_root is not None:
        document['runtime_source_root'] = str(Path(runtime_source_root).expanduser().resolve())
    _validate(document)
    return document


def capture_lerobot_camera_inputs(config, batch, *, runtime_source_root=None):
    """Capture native config order and actual batch key presence without reading pixels."""
    features = getattr(config, 'image_features', None)
    if not isinstance(features, dict) or not isinstance(batch, dict):
        raise InputError('Camera capture requires the resolved config.image_features and the actual batch dictionary')
    policy = getattr(config, 'type', None)
    expected = list(features)
    document = {'schema_version': 1, 'kind': KINDS[1], 'policy_type': policy,
                'expected_image_keys': expected, 'provided_image_keys': [key for key in expected if key in batch],
                'empty_cameras': getattr(config, 'empty_cameras', 0) if policy == 'smolvla' else None}
    if runtime_source_root is not None:
        document['runtime_source_root'] = str(Path(runtime_source_root).expanduser().resolve())
    _validate(document)
    return document


def _validate(document):
    if type(document) is not dict or document.get('kind') not in KINDS:
        raise InputError('Expected an openpi_initialization or lerobot_camera_inputs contract')
    if type(document.get('schema_version')) is not int or document['schema_version'] != 1:
        raise InputError('Pipeline contracts require schema_version=1')
    if 'runtime_source_root' in document:
        _text(document['runtime_source_root'], 'runtime_source_root', limit=4096)
        if '://' in document['runtime_source_root']:
            raise InputError('runtime_source_root must be local')
    if document['kind'] == KINDS[0]:
        required = {'kind', 'schema_version', 'backend', 'expected_initialization', 'train_config'}
        if not required <= set(document) or set(document) - required - {'runtime_source_root'}:
            raise InputError('Unexpected or missing fields in openpi_initialization contract')
        _text(document.get('backend'), 'backend', limit=32)
        if document.get('expected_initialization') not in ('pretrained', 'scratch', 'unspecified'):
            raise InputError('expected_initialization must be pretrained, scratch or unspecified')
        cfg = document.get('train_config')
        if type(cfg) is not dict or not {'name', 'resume', 'pytorch_weight_path', 'weight_loader'} <= cfg.keys():
            raise InputError('train_config needs the resolved name, resume, pytorch_weight_path and weight_loader fields')
        if set(cfg) != {'name', 'resume', 'pytorch_weight_path', 'weight_loader'}:
            raise InputError('train_config must contain only the captured initialization fields')
        _text(cfg['name'], 'train_config.name')
        if type(cfg['resume']) is not bool:
            raise InputError('train_config.resume must be a boolean')
        if cfg['pytorch_weight_path'] is not None:
            _text(cfg['pytorch_weight_path'], 'pytorch_weight_path', limit=4096)
        if type(cfg['weight_loader']) is not dict or type(cfg['weight_loader'].get('class')) is not str:
            raise InputError('weight_loader must name its resolved class')
        _text(cfg['weight_loader']['class'], 'weight_loader.class')
        if set(cfg['weight_loader']) - {'class', 'params_path'}:
            raise InputError('weight_loader contains unsupported fields')
        if cfg['weight_loader'].get('params_path') is not None:
            _text(cfg['weight_loader']['params_path'], 'weight_loader.params_path', limit=4096)
    else:
        required = {'kind', 'schema_version', 'policy_type', 'expected_image_keys', 'provided_image_keys', 'empty_cameras'}
        if not required <= set(document) or set(document) - required - {'runtime_source_root'}:
            raise InputError('Unexpected or missing fields in lerobot_camera_inputs contract')
        _text(document.get('policy_type'), 'policy_type', limit=64)
        expected = _keys(document.get('expected_image_keys'), 'expected_image_keys')
        provided = _keys(document.get('provided_image_keys'), 'provided_image_keys', empty=True)
        if not set(provided) <= set(expected):
            raise InputError('provided_image_keys must be selected from the declared expected_image_keys')
        if document['policy_type'] == 'smolvla':
            if type(document.get('empty_cameras')) is not int or not 0 <= document['empty_cameras'] <= 8:
                raise InputError('SmolVLA requires its resolved empty_cameras integer in [0,8]')
        elif document.get('empty_cameras') is not None:
            raise InputError('empty_cameras is only a SmolVLA setting; use null for other policies')


def check_pipeline_document(document, *, runtime_source_root=None):
    """Evaluate only the named source contract; other checks remain independent."""
    _validate(document)
    if runtime_source_root is None:
        runtime_source_root = document.get('runtime_source_root')
        if runtime_source_root is not None and not Path(runtime_source_root).is_absolute():
            raise InputError('A direct document check needs an absolute runtime_source_root or an explicit source directory')
    if document['kind'] == KINDS[0]:
        code = 'openpi.pretrained_load_selection'
        cfg = document['train_config']
        if document['backend'] != 'pytorch':
            return _report(document, 'unknown', code, 'This rule covers the OpenPI PyTorch trainer only; no JAX load verdict was inferred.',
                           'Check the actual backend and its initialization path.')
        profile, reason, evidence = _source(runtime_source_root, OPENPI_FILE, {OPENPI_HASH: (OPENPI_COMMIT, 442)})
        if profile is None:
            return _report(document, 'unknown', code, reason, 'Supply the actual supported OpenPI source; unknown code cannot approve initialization.', evidence=evidence)
        evidence.append({'path': 'pipeline_contract', 'pointer': '/train_config/pytorch_weight_path',
                         'value': cfg['pytorch_weight_path']})
        if cfg['resume']:
            return _report(document, 'unknown', code, 'Resume is selected; this fresh-initialization check does not verify the restored checkpoint.',
                           'Verify the actual resume checkpoint and restore result.', evidence=evidence, profile=profile)
        selected = cfg['pytorch_weight_path'] is not None
        intent = document['expected_initialization']
        if not selected and intent == 'pretrained':
            status, message = 'error', 'Pretrained loading is skipped: pytorch_weight_path=null. This PyTorch trainer does not use weight_loader.'
            remedy = 'Set the resolved TrainConfig.pytorch_weight_path to the converted PyTorch checkpoint directory, then recapture and recheck before training.'
        elif not selected and intent == 'unspecified':
            status, message = 'warning', 'The PyTorch pretrained-load branch is skipped; initialization intent was not declared.'
            remedy = 'Declare pretrained or intentional scratch initialization, and check the actual resolved config.'
        elif selected and intent == 'scratch':
            status, message = 'error', 'A pretrained-load path is selected although scratch initialization was explicitly requested.'
            remedy = 'Resolve the intended initialization; for scratch, remove the PyTorch checkpoint path and recapture the config.'
        else:
            status = 'pass'
            message = ('Pretrained-load branch: selected. The trainer will attempt model.safetensors under the configured path; file contents and model compatibility were not checked.'
                       if selected else 'Pretrained-load branch: intentionally skipped for declared scratch training.')
            remedy = 'Keep the resolved initialization settings; separately verify checkpoint availability, loading and model compatibility.'
        return _report(document, status, code, message, remedy, evidence=evidence, profile=profile,
                       pretrained_load_selected=selected, expected_initialization=intent,
                       weight_file_checked=False, weight_loader_used_by_this_branch=False)
    code = 'lerobot.camera_slot_order'
    policy = document['policy_type']
    if policy not in CAMERA_PROFILES:
        return _report(document, 'unknown', code, f'Camera slot behavior for {policy} is outside this rule.',
                       'Use the policy-specific native preprocessor; no FastWAM or generic camera assumption was applied.')
    relative, hashes, _ = CAMERA_PROFILES[policy]
    profile, reason, evidence = _source(runtime_source_root, relative, hashes)
    if profile is None:
        return _report(document, 'unknown', code, reason, 'Supply the actual supported LeRobot source or installed package directory.', evidence=evidence)
    expected, supplied = document['expected_image_keys'], set(document['provided_image_keys'])
    present = [key for key in expected if key in supplied]
    missing = [key for key in expected if key not in supplied]
    if not present:
        return _report(document, 'error', code, 'No declared camera is present; the native preprocessor would reject this batch.',
                       'Restore at least one genuine camera input; do not synthesize an unmasked replacement.', evidence=evidence, profile=profile)
    pads = min(len(missing), document['empty_cameras']) if policy == 'smolvla' else len(missing)
    observed = present + [None] * pads
    shifts = [{'camera': key, 'expected_slot': expected.index(key), 'actual_slot': index}
              for index, key in enumerate(present) if expected.index(key) != index]
    evidence.append({'path': 'pipeline_contract', 'pointer': '/provided_image_keys', 'value': document['provided_image_keys']})
    if shifts:
        first = shifts[0]
        message = (f"{first['camera']} moves from its declared slot {first['expected_slot']} to slot {first['actual_slot']}; "
                   f'{len(shifts)} present camera(s) shift because missing cameras are padded after present ones.')
        remedy = ('Restore the genuine missing camera under its trained key, recapture the actual batch and recheck. '
                  'For intentional ablation, use a separately verified order-preserving native preprocessor. '
                  'Never insert an unmasked fake camera.')
    else:
        message = 'Present camera slots: consistent. Missing trailing cameras retain native padding/omission behavior; camera adequacy and image contents were not checked.'
        remedy = 'Keep the declared camera order; check this guard on each new batch to catch later sensor dropout.'
    return _report(document, 'error' if shifts else 'pass', code, message, remedy,
                   evidence=evidence, profile=profile, declared_slots=expected, native_slot_cameras=observed,
                   shifted_cameras=shifts, missing_image_keys=missing)


def check_pipeline_contract(source, *, runtime_source_root=None, authorize=None):
    if not isinstance(source, (str, os.PathLike)):
        raise InputError('Pipeline contract source must be a local file path')
    name = Path(source).name
    _text(name, 'pipeline contract filename')
    raw, _ = _read_small(source, limit=256 * 1024)
    document = _reference_json(raw)
    _validate(document)
    if runtime_source_root is None:
        runtime_source_root = document.get('runtime_source_root')
        if runtime_source_root is not None:
            runtime_source_root = Path(runtime_source_root).expanduser()
            if not runtime_source_root.is_absolute():
                runtime_source_root = Path(source).resolve().parent / runtime_source_root
    if runtime_source_root is not None and authorize is not None:
        runtime_source_root = authorize(str(runtime_source_root))
    report = check_pipeline_document(document, runtime_source_root=runtime_source_root)
    report['contract_sha256'] = hashlib.sha256(raw).hexdigest()
    report['contract_file'] = name
    return report


class PipelineContractError(InputError):
    def __init__(self, report):
        self.report = report
        super().__init__(report['findings'][0]['message'])


def guard_openpi_initialization(config, *, runtime_source_root=None, expected_initialization='pretrained'):
    """Call immediately before the existing PyTorch trainer constructs its model."""
    document = capture_openpi_initialization(config, expected_initialization=expected_initialization,
                                              runtime_source_root=runtime_source_root)
    report = check_pipeline_document(document, runtime_source_root=runtime_source_root)
    if report['status'] != 'pass':
        raise PipelineContractError(report)
    return report


def guard_lerobot_camera_inputs(config, batch, *, runtime_source_root):
    """Call before select_action/forward, on each batch that can lose a camera."""
    document = capture_lerobot_camera_inputs(config, batch)
    report = check_pipeline_document(document, runtime_source_root=runtime_source_root)
    if report['status'] != 'pass':
        raise PipelineContractError(report)
    return report


def _camera_code_signature(code):
    """Compare active method semantics without filenames/line-table bookkeeping."""
    constants = tuple(_camera_code_signature(value) if isinstance(value, CodeType)
                      else (type(value).__name__, repr(value)) for value in code.co_consts)
    return (code.co_code, constants, code.co_names, code.co_varnames,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_freevars, code.co_cellvars, code.co_flags & 0xFFFF,
            getattr(code, 'co_exceptiontable', b''))


def _verify_camera_method_body(method, source, profile, method_name):
    """Compile audited text without importing it and bind the active method body."""
    with open_regular(source, 'native camera method source') as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != profile['sha256']:
        raise InputError('The native camera source changed during guard installation')
    try:
        module = compile(raw, str(source), 'exec', dont_inherit=True)
    except (SyntaxError, ValueError, TypeError) as error:
        raise InputError('The audited camera source cannot be compiled on this interpreter') from error
    pending, candidates = [module], []
    while pending:
        code = pending.pop()
        if code.co_name == method_name:
            candidates.append(code)
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
    active = method.__func__
    if (len(candidates) != 1 or active.__name__ != method_name
            or _camera_code_signature(active.__code__) != _camera_code_signature(candidates[0])):
        raise InputError('The active camera method body differs from the audited native method')


class CameraInputGuard:
    """A client-side hook at the real native preprocessing boundary.

    No method is invoked by installation or by MCP. The existing application
    invokes its policy normally; an unsafe batch stops before pixel processing.
    """

    def __init__(self, policy, *, runtime_source_root=None):
        if getattr(policy, '_roboclue_camera_input_guard', None) is not None:
            raise InputError('A camera-input guard is already installed on this policy')
        self.policy = policy
        self.last_contract = None
        self.last_report = None
        prototype = capture_lerobot_camera_inputs(policy.config, dict.fromkeys(policy.config.image_features))
        if prototype['policy_type'] not in CAMERA_PROFILES:
            raise InputError('This camera-input guard supports pi0, pi05 and smolvla only')
        self.method_name = CAMERA_PROFILES[prototype['policy_type']][2]
        self.original = getattr(policy, self.method_name, None)
        if not inspect.ismethod(self.original):
            raise InputError('The policy must expose its original native bound camera-preprocessing method')
        if self.original.__self__ is not policy:
            raise InputError('The camera-preprocessing method is bound to a different policy instance')
        actual_file = inspect.getsourcefile(self.original)
        if actual_file is None:
            raise InputError('The active camera method has no inspectable source file')
        if runtime_source_root is None:
            runtime_source_root = Path(actual_file).resolve().parents[2]
        self.runtime_source_root = runtime_source_root
        ready = check_pipeline_document(prototype, runtime_source_root=runtime_source_root)
        if ready['status'] != 'pass':
            raise PipelineContractError(ready)
        expected_file = contained(Path(runtime_source_root).expanduser().resolve(), ready['source_profile']['file'])
        if actual_file is None or Path(actual_file).resolve() != expected_file:
            raise InputError('The active camera method does not originate from the supplied profiled source file')
        _verify_camera_method_body(self.original, expected_file, ready['source_profile'], self.method_name)
        self.original_code = self.original.__func__.__code__

        def checked(instance, batch, *args, **kwargs):
            if (instance is not self.policy or not inspect.ismethod(self.original)
                    or self.original.__self__ is not self.policy
                    or self.original.__func__.__code__ is not self.original_code):
                raise InputError('The guarded camera method binding or body changed after installation')
            self.last_contract = capture_lerobot_camera_inputs(instance.config, batch,
                                                                 runtime_source_root=self.runtime_source_root)
            self.last_report = check_pipeline_document(self.last_contract, runtime_source_root=self.runtime_source_root)
            if self.last_report['status'] != 'pass':
                raise PipelineContractError(self.last_report)
            return self.original(batch, *args, **kwargs)

        self.bound = MethodType(checked, policy)
        setattr(policy, self.method_name, self.bound)
        policy._roboclue_camera_input_guard = self

    def remove(self):
        if getattr(self.policy, self.method_name) != self.bound:
            raise InputError('The camera method changed after guard installation; refusing to overwrite it')
        setattr(self.policy, self.method_name, self.original)
        delattr(self.policy, '_roboclue_camera_input_guard')


def install_lerobot_camera_guard(policy, *, runtime_source_root=None):
    """Install a fail-closed guard; it never fabricates, renames or reorders images."""
    return CameraInputGuard(policy, runtime_source_root=runtime_source_root)


def render_pipeline_contract(report):
    finding = report['findings'][0]
    label = 'OpenPI pretrained-load selection' if report['contract_kind'] == KINDS[0] else 'LeRobot camera-slot order'
    verdict = {'pass': 'CONSISTENT', 'error': 'ERROR', 'warning': 'WARNING', 'unknown': 'NOT CHECKED'}[report['status']]
    profile = report.get('source_profile')
    location = (f"{profile['file']}:{profile['line']} (file hash matches {profile['revision'][:8]})"
                if profile else 'Source behavior is unassessed; inspect the supplied file/profile.')
    if profile and profile['revision'].startswith('pypi:'):
        location = f"{profile['file']}:{profile['line']} (file hash matches {profile['revision']})"
    observed = next((item for item in finding['evidence'] if item.get('path') == 'pipeline_contract'), None)
    if observed:
        value = json.dumps(observed.get('value'), ensure_ascii=False)
        if len(value) > 160:
            value = '(value exceeds 160 characters; see JSON)'
        location = f"{report.get('contract_file', 'resolved runtime fields')}#{observed['pointer']}={value}; " + location
    return (f'Verdict: {verdict} — {label} only.\nDetails: {finding["message"]}\n'
            f'Evidence: {location}\nNext: {finding["remedy"]}\n'
            'Scope: declared fields/source; active loading, model execution and robot success are unverified.\n')


def concise_pipeline_contract(report):
    result = json.loads(json.dumps(report, allow_nan=False))
    # Preserve the full evidence for these small contracts; refuse growth rather
    # than silently cutting camera identities or losing an ERROR.
    envelope = {'content': [{'type': 'text', 'text': render_pipeline_contract(result).rstrip()}],
                'structuredContent': result, 'isError': False}
    if len(json.dumps(envelope, allow_nan=False).encode()) > 8 * 1024:
        raise InputError('The pipeline-contract preview exceeds 8 KiB including text; use shorter camera keys and the CLI JSON report.')
    return result
