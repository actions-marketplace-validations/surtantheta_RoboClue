"""Static, exact-source FastWAM contracts; never imports the inspected runtime."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import numpy as np

from .artifacts import InputError, load_checkpoint
from .fs_guard import open_regular, refuse_special_files

COMMIT = '77710dee471d5cfb070fad1adc30c8f94dc14757'
TARBALL_SHA256 = '36067643cfe78a3efb7f50926cc6c240d9618239b51dc1850c5fd7de27018ba7'
PROFILES = json.loads(Path(__file__).with_name('fastwam_profiles.json').read_text())
SOURCE_FILES = PROFILES['main-77710dee']['files']
ATOL = 1e-6
SCOPE = ('Saved-processor inference/eval image map, or explicitly selected fresh-training config. Resume overrides are unassessed. '
         'Static files/source only; active runtime, native loading, model execution, other modalities '
         'and task success are not established.')


def _result(status, message, *, evidence=None, images=None, remedy=None):
    return {'kind': 'fastwam_image_contract', 'rule_id': 'RC009', 'contract_id': 'WAM-FW-01',
            'status': status, 'scope': SCOPE,
            'findings': [{'code': 'fastwam.image_map', 'status': status, 'message': message,
                          'evidence': evidence or [], 'remedy': remedy or
                          'Supply the actual runtime_source_root (CLI --runtime-source-root) or inspect the named unsupported input; unknown source remains unassessed.'}],
            'images': images or {}, 'expected_map': {'slope': 2.0, 'offset': -1.0, 'atol': ATOL},
            'input_assumption': 'float32 RGB pixels in [0,1], before preprocessing',
            'active_runtime': 'not_attested', 'native_execution': 'not_run'}


def discover_source_root():
    """Locate installed source using distribution metadata, without importing LeRobot."""
    try:
        return Path(importlib.metadata.distribution('lerobot').locate_file('')).resolve()
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_profile(root):
    try:
        root = Path(root).expanduser().resolve()
    except (OSError, RuntimeError) as error:
        raise InputError('Cannot resolve runtime_source_root; inspect symlink loops and filesystem errors.') from error
    if not root.is_dir():
        raise InputError('runtime_source_root must be a local LeRobot checkout or site-packages directory')
    package = next((p for p in (root / 'src/lerobot', root / 'lerobot', root)
                    if (p / 'policies/fastwam').is_dir()), None)
    if package is None:
        return None, [], 'No src/lerobot, lerobot, or package-root FastWAM source layout found.'
    mismatches = []
    for label, profile in PROFILES.items():
        evidence = []
        for relative, expected in profile['files'].items():
            path = package / relative
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError) as error:
                raise InputError('Cannot resolve FastWAM source file: ' + relative) from error
            if not resolved.is_relative_to(root):
                raise InputError('FastWAM source file escapes runtime_source_root: ' + relative)
            try:
                with open_regular(path, 'FastWAM source') as stream:
                    content = stream.read(2 * 1024 * 1024 + 1)
            except FileNotFoundError:
                mismatches.append(label + ': missing ' + relative)
                break
            digest = hashlib.sha256(content).hexdigest()
            if len(content) > 2 * 1024 * 1024 or digest != expected:
                mismatches.append(label + ': mismatch ' + relative + ' sha256=' + digest)
                break
            evidence.append({'path': str(path), 'sha256': digest})
        else:
            return dict(profile, name=label), evidence, None
    return None, [], '; '.join(mismatches)


def _channel_stat(value):
    array = np.asarray(value, dtype=np.float32)
    if array.shape not in ((3,), (3, 1, 1)) or not np.isfinite(array).all():
        raise ValueError('Only finite RGB per-channel statistics are supported')
    return array.reshape(3)


def _device_declaration(value, *, optional):
    """Bound the supported serialized syntax, without checking hardware availability."""
    if value is None:
        return optional
    return (type(value) is str and len(value) <= 64
            and re.fullmatch(r'(?:cpu|cuda|mps|xpu)(?::[0-9]{1,5})?', value) is not None)


def _preview_text(value, limit):
    if len(value) <= limit:
        return value, None
    detail = {'characters': len(value), 'sha256': hashlib.sha256(value.encode()).hexdigest()}
    return value[:limit] + ' [truncated; complete value is in the detailed report]', detail


def check_image_contract(checkpoint, runtime_source_root=None, runtime_mode='saved', *, authorize_source=None):
    """Return one scoped contract, never substituting for generic checkpoint audit."""
    if runtime_mode not in ('saved', 'training'):
        raise InputError('runtime_mode must be saved or training')
    root = Path(checkpoint).expanduser().resolve()
    if root.is_file():
        root = root.parent
    refuse_special_files(root, 'checkpoint')
    cp = load_checkpoint(root)
    if cp.config.get('type') != 'fastwam':
        return None
    features = cp.config.get('input_features')
    if not isinstance(features, dict):
        return _result('unknown', 'FastWAM image map not checked: input_features must be an object.')
    visuals = {key: value for key, value in features.items()
               if isinstance(value, dict) and value.get('type') == 'VISUAL'}
    if not visuals or len(visuals) > 8:
        return _result('unknown', 'FastWAM image map not checked: one to eight declared image features are required.')
    if any(len(key) > 256 or not key.startswith('observation.images.') or value.get('shape', [])[:1] != [3]
           or len(value.get('shape', [])) != 3 for key, value in visuals.items()):
        return _result('unknown', 'FastWAM image map not checked: unsupported RGB image layout.')
    discovered = runtime_source_root is None
    if discovered:
        runtime_source_root = discover_source_root()
    if runtime_source_root is None:
        return _result('unknown', 'FastWAM image map not checked: no supplied or statically discoverable installed source.')
    if authorize_source is not None:
        try:
            runtime_source_root = authorize_source(str(runtime_source_root))
        except InputError:
            if not discovered:
                raise
            return _result('unknown', 'FastWAM image map not checked: automatically discovered installed source could not be authorized under the MCP read roots.',
                           remedy='Add the actual installed source directory to the MCP read roots or supply an explicitly authorized runtime_source_root; the generic checkpoint audit is retained.')
    profile, evidence, reason = _source_profile(runtime_source_root)
    if profile is None:
        return _result('unknown', 'FastWAM image map not checked: ' + reason)
    expected_steps = ['rename_observations_processor', 'to_batch_processor', 'device_processor', 'normalizer_processor']
    if [step['name'] for step in cp.pre] != expected_steps:
        return _result('unknown', 'FastWAM image map not checked: unsupported saved processor order.', evidence=evidence)
    if any(step.get('missing_state') for step in cp.pre + cp.post):
        return _result('unknown', 'FastWAM image map not checked: a declared processor state file is missing.', evidence=evidence)
    rename, batch, device, normalizer = [step['config'] for step in cp.pre]
    if (not _device_declaration(device.get('device', 'cpu'), optional=False)
            or not _device_declaration(normalizer.get('device'), optional=True)):
        return _result('unknown', 'FastWAM image map not checked: unsupported serialized processor device declaration; device availability was not checked.', evidence=evidence)
    if (set(rename) - {'rename_map'} or not isinstance(rename.get('rename_map', {}), dict)
            or rename.get('rename_map', {}) or batch or set(device) - {'device', 'float_dtype'}
            or device.get('float_dtype') not in (None, 'float32')
            or set(normalizer) - {'eps', 'features', 'norm_map', 'normalize_observation_keys', 'stats', 'device', 'dtype'}
            or normalizer.get('stats') or normalizer.get('dtype') not in (None, 'float32')):
        return _result('unknown', 'FastWAM image map not checked: unsupported renaming, dtype or processor settings.', evidence=evidence)
    if any(normalizer.get('features', {}).get(key) != feature for key, feature in visuals.items()):
        return _result('unknown', 'FastWAM image map not checked: processor image declarations differ.', evidence=evidence)
    selected = normalizer.get('normalize_observation_keys')
    if selected is not None and (not isinstance(selected, list) or any(not isinstance(k, str) for k in selected)):
        return _result('unknown', 'FastWAM image map not checked: unsupported normalization key selection.', evidence=evidence)
    mapping = normalizer.get('norm_map', {})
    if any(key not in ('VISUAL', 'STATE', 'ACTION', 'ENV') for key in mapping):
        return _result('unknown', 'FastWAM image map not checked: invalid native norm_map feature type.', evidence=evidence)
    mode = mapping.get('VISUAL', 'IDENTITY')
    training_map = cp.config.get('normalization_mapping')
    training_visual = training_map.get('VISUAL', 'IDENTITY') if isinstance(training_map, dict) else None
    if runtime_mode == 'training':
        # Native trainer replaces saved norm_map with policy.config mapping and,
        # for fresh fine-tuning, saved statistics with dataset.meta.stats.
        if training_visual != 'IDENTITY':
            return _result('unknown', 'Fresh-training image map not checked: config.json normalization_mapping.VISUAL=' +
                           str(training_visual)[:80] + '; native fine-tuning replaces saved stats with dataset stats.',
                           evidence=evidence, remedy='Set and verify VISUAL=IDENTITY in both config.json normalization_mapping and policy_preprocessor.json norm_map in an owned copy; otherwise capture the effective dataset statistics before assessing fresh training.')
        mode = 'IDENTITY'
    if mode not in ('IDENTITY', 'MEAN_STD'):
        return _result('unknown', 'FastWAM image map not checked: unsupported visual normalization mode.', evidence=evidence)
    eps = normalizer.get('eps', 1e-8)
    if type(eps) not in (int, float) or not 0 <= eps <= 1:
        return _result('unknown', 'FastWAM image map not checked: unsupported epsilon.', evidence=evidence)
    maps = {}
    step = cp.pre[-1]
    for key in visuals:
        slope, offset = np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        active = mode != 'IDENTITY' and (selected is None or key in selected)
        saved = {'mode': mode, 'active': active, 'file': step['source'], 'pointer': step['pointer'] + '/config/norm_map/VISUAL'}
        if active:
            stats = step['stats'].get(key, {})
            if not {'mean', 'std'} <= set(stats):
                return _result('unknown', 'FastWAM image map not checked: saved image statistics are missing.', evidence=evidence)
            try:
                mean, std = _channel_stat(stats['mean']), _channel_stat(stats['std'])
                denom = std + np.float32(eps)
                if np.any(std < 0) or np.any(denom <= 0):
                    raise ValueError('Invalid standard deviation')
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    slope, offset = 1 / denom, -mean / denom
            except (ValueError, TypeError, OverflowError, FloatingPointError):
                return _result('unknown', 'FastWAM image map not checked: invalid per-channel statistics.', evidence=evidence)
            saved.update(file=step.get('state_file'), mean_key=key + '.mean', std_key=key + '.std',
                         mean=mean.tolist(), std=std.tolist(), eps=eps)
        with np.errstate(over='ignore', invalid='ignore'):
            slope, offset = slope * 2, offset * 2 - 1
        if not np.isfinite(slope).all() or not np.isfinite(offset).all():
            return _result('unknown', 'FastWAM image map not checked: numeric range exceeded.', evidence=evidence)
        maps[key] = {'slope': slope.tolist(), 'offset': offset.tolist(), 'saved': saved,
                     'consistent': bool(np.allclose(slope, 2, atol=ATOL, rtol=0) and np.allclose(offset, -1, atol=ATOL, rtol=0))}
    evidence.extend({'path': a.path, 'sha256': a.sha256, 'role': a.role}
                    for a in cp.artifacts if a.role in ('checkpoint_config', 'preprocessor', 'preprocessor_state'))
    broken = [key for key, value in maps.items() if not value['consistent']]
    if broken:
        message = 'FastWAM composed image map deviates from the model input 2*x-1 for: ' + ', '.join(broken)
        remedy = 'Test VISUAL=IDENTITY in both policy_preprocessor.json norm_map and config.json normalization_mapping in an owned copy; preserve state/action statistics and rerun the same source/mode check.'
    else:
        message = 'FastWAM image map: consistent with 2*x-1 in ' + runtime_mode + ' mode. Other modalities and native execution were not checked.'
        remedy = 'Retain this scoped image pairing; inspect all independent checkpoint findings. Fresh training must also verify config.json normalization_mapping and effective dataset statistics.'
    report = _result('error' if broken else 'pass', message, evidence=evidence, images=maps, remedy=remedy)
    report['runtime_mode'] = runtime_mode
    if isinstance(training_visual, str):
        report['training_visual_mode'], truncation = _preview_text(training_visual, 80)
    else:
        rendered = json.dumps(training_visual, sort_keys=True, allow_nan=False)
        preview, truncation = _preview_text(rendered, 80)
        report['training_visual_mode'] = training_visual if truncation is None else preview
    if truncation is not None:
        report['training_visual_mode_truncation'] = truncation
    report['runtime_profile'] = {'profile': profile['name'], 'source_identity': profile['identity'],
                                 'selection': 'exact hashes of all profiled processor, FastWAM, config and construction files',
                                 'contract_source': ['policies/fastwam/wan/modular.py:' + str(line) for line in profile['map_lines']]}
    return report


def augment_checkpoint_report(report, image_report):
    """Retain every generic finding and its gate, then add the independent image check."""
    if image_report is None:
        return report
    report['image_contract'] = image_report
    report['findings'].extend(image_report['findings'])
    report['counts'] = {s: sum(f['status'] == s for f in report['findings']) for s in ('error', 'unknown', 'warning', 'pass')}
    report['status'] = next((s for s in ('error', 'unknown', 'warning') if report['counts'][s]), 'pass')
    return report


def evidence_line(report):
    pieces = []
    for key, image in report.get('images', {}).items():
        saved = image['saved']
        values = ('mean=' + str(saved['mean']) + ',std=' + str(saved['std'])) if 'mean' in saved else 'VISUAL=' + str(saved['mode'])
        pieces.append(key + ': ' + Path(saved['file']).name + ' ' + values + ' → slope=' + str(image['slope']) + ',offset=' + str(image['offset']))
    source = ','.join(report.get('runtime_profile', {}).get('contract_source', []))
    return '; '.join(pieces) + ('; ' + source if source else '')


def concise_image_contract(report):
    result = {key: value for key, value in report.items() if key != 'findings'}
    finding = report['findings'][0]
    result['findings'] = [{key: value for key, value in finding.items() if key != 'evidence'}]
    result['evidence_files'] = len(finding['evidence'])
    result['full_report_sha256'] = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result


def concise_checkpoint_report(report):
    """Keep every finding code/status; replace repeated full processor values with references."""
    import copy

    result = {key: copy.deepcopy(report[key]) for key in
              ('kind', 'schema_version', 'status', 'counts', 'scope', 'assessment_scope') if key in report}
    result['image_contract'] = copy.deepcopy(concise_image_contract(report['image_contract']))
    result['findings'] = []
    for finding in report['findings']:
        item = {key: finding[key] for key in ('code', 'status')}
        # The RC009 finding lives once under image_contract; its code/status still
        # participates in the full checkpoint finding list and counts.
        if finding['code'] != 'fastwam.image_map':
            item['message'], truncation = _preview_text(finding.get('message', ''), 1024)
            if truncation is not None:
                item['message_truncation'] = truncation
            if finding.get('remedy') and finding['status'] == 'error':
                item['remedy'], truncation = _preview_text(finding['remedy'], 1024)
                if truncation is not None:
                    item['remedy_truncation'] = truncation
            if finding['status'] == 'error' and finding.get('evidence'):
                item['evidence'] = [{key: value for key, value in finding['evidence'][0].items()
                                     if key in ('path', 'pointer', 'sha256')}]
        result['findings'].append(item)
    result['detail_omissions'] = 'Repeated processor values, artifact inventories and full finding evidence omitted; all finding codes/statuses and counts retained. Any truncated text is explicitly marked with its original length and digest. Repeat response_format=detailed for a fresh full read.'
    result['full_report_sha256'] = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    image = result['image_contract']

    def over_budget():
        return len(json.dumps(result, allow_nan=False, indent=2).encode()) > 8 * 1024

    # Supported camera keys can each be 256 characters. Prefer an explicit
    # omitted-camera count over repeating eight large numeric/path witnesses.
    # Keep an inconsistent map first when independent cameras differ.
    if over_budget() and image.get('images'):
        original_images = image['images']
        image['images'] = dict(sorted(original_images.items(), key=lambda pair: pair[1]['consistent']))
        image['images_total'] = len(original_images)
        image['image_status_counts'] = {
            'consistent': sum(row['consistent'] for row in original_images.values()),
            'inconsistent': sum(not row['consistent'] for row in original_images.values()),
        }
        image['full_images_sha256'] = hashlib.sha256(
            json.dumps(original_images, sort_keys=True, allow_nan=False).encode()).hexdigest()
        image['images_omitted'] = 0
        while len(image['images']) > 1 and over_budget():
            image['images'].pop(next(reversed(image['images'])))
            image['images_omitted'] += 1
    if over_budget():
        for key, row in image.get('images', {}).items():
            saved = row['saved']
            if saved.get('mean_key') == key + '.mean' and saved.get('std_key') == key + '.std':
                saved.pop('mean_key')
                saved.pop('std_key')
                saved['statistic_keys'] = {'base': 'image_key', 'mean_suffix': '.mean', 'std_suffix': '.std'}
    if over_budget():
        for finding in image['findings']:
            for field, limit in (('message', 320), ('remedy', 512)):
                text, truncation = _preview_text(finding.get(field, ''), limit)
                finding[field] = text
                if truncation is not None:
                    finding[field + '_truncation'] = truncation
    return result
