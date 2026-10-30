"""Capture trusted, already-loaded pipelines; never import code from artifacts.

This Python-only writer is intentionally absent from the read-only MCP server.
Call after runtime overrides/reconciliation, while configuration is quiescent.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from . import __version__
from .artifacts import (
    MAX_JSON_BYTES,
    MAX_STATS_BYTES,
    InputError,
    contained,
    finite_tree,
    load_checkpoint,
    read_json,
    sha256,
    step_kind,
)
from .diagnostics import audit
from .publication import publish_capture

STATS = {'mean', 'std', 'min', 'max', 'q01', 'q99', 'q10', 'q90', 'count', 'q50'}
STANDARD_CONVERTERS = {f'lerobot.processor.converters.{name}' for name in (
    'batch_to_transition', 'transition_to_batch', 'policy_action_to_transition', 'transition_to_policy_action')}


def _json(value: Any) -> bytes:
    try:
        finite_tree(value)
        data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise InputError(f'Capture requires finite JSON configuration: {exc}') from exc
    if not isinstance(value, dict) or len(data) > MAX_JSON_BYTES:
        raise InputError('Capture configuration must be a bounded JSON object')
    return data


def _identity(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, '__module__', type(value).__module__)) + '.' + str(
        getattr(value, '__qualname__', type(value).__qualname__))


def _snapshot(pipeline: Any) -> tuple[dict, dict, dict]:
    if not callable(getattr(pipeline, 'get_config', None)) or not callable(getattr(pipeline, 'state_dict', None)):
        raise InputError('Loaded pipeline must expose get_config() and state_dict()')
    config = json.loads(_json(pipeline.get_config()))
    steps = config.get('steps')
    if not isinstance(steps, list) or len(steps) > 100:
        raise InputError('Pipeline steps must be a list with at most 100 entries')
    raw_states = pipeline.state_dict()
    if not isinstance(raw_states, dict):
        raise InputError('Pipeline state_dict must group tensors by suffixless state key')
    states, used, total_bytes, total_values = {}, set(), 0, 0
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get('config', {}), dict):
            raise InputError('Malformed captured processor step')
        state_file = step.get('state_file')
        if not state_file:
            continue
        if not isinstance(state_file, str) or Path(state_file).name != state_file or not state_file.endswith('.safetensors'):
            raise InputError('Captured state references must be local safetensors filenames')
        key = state_file[:-12]
        if step_kind(step.get('registry_name') or step.get('class', '')) not in ('normalize', 'unnormalize'):
            raise InputError('Only normalization statistics may be captured; custom states/weights are unsupported')
        state = raw_states.get(key)
        if not isinstance(state, dict) or not state:
            raise InputError('Captured normalizer references missing statistics state')
        arrays = {}
        for name, value in state.items():
            if not isinstance(name, str) or name.rpartition('.')[2] not in STATS:
                raise InputError('Only named normalization statistics may be captured')
            # Torch is optional: the caller already trusts these in-process objects.
            if hasattr(value, 'detach'):
                try:
                    value = value.detach().cpu().numpy()
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise InputError(f'Unsupported statistics tensor dtype/device; no implicit precision conversion: {exc}') from exc
            array = np.asarray(value)
            total_values += array.size
            total_bytes += array.nbytes
            if total_values > 1_000_000 or total_bytes > MAX_STATS_BYTES:
                raise InputError('Captured statistics exceed the value/byte budget')
            if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
                raise InputError('Captured statistics must be finite numeric tensors')
            arrays[name] = np.array(array, copy=True, order='C')
        states[key] = arrays
        used.add(key)
    if set(raw_states) != used:
        raise InputError('Unreferenced pipeline states are unsupported; capture only normalization statistics')
    converters = {key: _identity(getattr(pipeline, key, None)) for key in ('to_transition', 'to_output')}
    for key in ('before_step_hooks', 'after_step_hooks'):
        hooks = getattr(pipeline, key, [])
        if not isinstance(hooks, (list, tuple)):
            raise InputError('Pipeline hooks must be a sequence')
        converters[key] = [_identity(hook) for hook in hooks]
    return config, states, converters


def _fingerprint(snapshot: tuple[dict, dict, dict]) -> str:
    config, states, converters = snapshot
    digest = hashlib.sha256(_json({'config': config, 'converters': converters}))
    for key, arrays in sorted(states.items()):
        for name, array in sorted(arrays.items()):
            digest.update(_json({'state': key, 'name': name, 'dtype': array.dtype.str, 'shape': list(array.shape)}))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _relative_bindings(pipelines: dict, snapshots: dict) -> list[dict]:
    """Record actual object topology; equal-looking configurations are not linkage."""
    pre = getattr(pipelines['pre'], 'steps', None)
    post = getattr(pipelines['post'], 'steps', None)
    enabled = {i for i, s in enumerate(snapshots['pre'][0]['steps'])
               if step_kind(s.get('registry_name') or s.get('class', '')) == 'relative' and s.get('config', {}).get('enabled')}
    bindings = []
    for index, config in enumerate(snapshots['post'][0]['steps']):
        if step_kind(config.get('registry_name') or config.get('class', '')) != 'absolute' or not config.get('config', {}).get('enabled'):
            continue
        linked, status = None, 'not_inspected'
        if isinstance(pre, (list, tuple)) and isinstance(post, (list, tuple)) and index < len(post):
            reference = getattr(post[index], 'relative_step', None)
            if reference is None:
                status = 'missing'
            else:
                linked = next((i for i, step in enumerate(pre) if step is reference), None)
                status = 'paired' if linked in enabled else 'external'
        bindings.append({'post_step': index, 'pre_step': linked, 'status': status})
    return bindings


def capture_runtime(output: str | Path, *, policy_config: dict, preprocessor: Any,
                    postprocessor: Any, deployment: dict | None = None) -> dict:
    """Export effective loaded configuration/stats, without calling the policy.

    `policy_config` must describe the running policy, including its choice/type key.
    Converters are recorded by identity only. No units or live-driver facts are inferred.
    """
    destination = Path(output).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise InputError('Capture destination already exists; choose a new directory')
    config_bytes = _json(copy.deepcopy(policy_config))
    deployment_bytes = _json(copy.deepcopy(deployment)) if deployment is not None else None
    pipelines = {'pre': preprocessor, 'post': postprocessor}
    snapshots = {side: _snapshot(pipeline) for side, pipeline in pipelines.items()}
    bindings = _relative_bindings(pipelines, snapshots)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.policy-doctor-capture-', dir=destination.parent))
    try:
        checkpoint = staging / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'config.json').write_bytes(config_bytes)
        converters = {}
        for side, snapshot in snapshots.items():
            config, states, converters[side] = snapshot
            config = copy.deepcopy(config)
            for index, step in enumerate(config['steps']):
                if step.get('state_file'):
                    arrays = states[step['state_file'][:-12]]
                    step['state_file'] = f'{side}_{index:03d}.safetensors'
                    save_file(arrays, str(checkpoint / step['state_file']))
            (checkpoint / f'policy_{side}processor.json').write_bytes(_json(config))
        if deployment_bytes is not None:
            (staging / 'deployment.json').write_bytes(deployment_bytes)
        for side, pipeline in pipelines.items():
            if _fingerprint(snapshots[side]) != _fingerprint(_snapshot(pipeline)):
                raise InputError('Loaded pipeline changed during capture; quiesce it and retry')
        if bindings != _relative_bindings(pipelines, snapshots):
            raise InputError('Relative processor binding changed during capture')
        versions = {}
        for name in ('policy-deployment-doctor', 'lerobot', 'torch', 'numpy', 'safetensors'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        versions['policy-deployment-doctor'] = __version__
        files = [{'path': p.relative_to(staging).as_posix(), 'sha256': sha256(p)}
                 for p in sorted(staging.rglob('*')) if p.is_file()]
        manifest = {'schema_version': 1, 'engine_version': __version__,
                    'captured_at': datetime.now(timezone.utc).isoformat(),
                    'stage': 'caller-supplied loaded objects after overrides/reconciliation',
                    'versions': versions, 'converters': converters, 'files': files,
                    'relative_bindings': bindings,
                    'has_deployment': deployment_bytes is not None,
                    'scope': 'Loaded processor configs/statistics; converter identities only. No model weights, inference, anchor cache or driver measurements.'}
        (staging / 'capture.json').write_bytes(_json(manifest))
        (staging / 'READY.json').write_bytes(_json({'schema_version': 1,
                                                  'manifest_sha256': sha256(staging / 'capture.json')}))
        audit_capture(str(staging))
        publish_capture(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # Return paths from the final directory, not temporary staging paths.
    return audit_capture(str(destination))


def audit_capture(capture: str, dataset: str | None = None, *, expected_names=None,
                  expected_frame=None, expected_quantity=None, expected_identity=None,
                  atol=1e-6, rtol=1e-5, max_query_gap=None, command_contract=None, named_contract=None) -> dict:
    """Verify a local snapshot's consumed bytes and re-audit with this engine."""
    if command_contract is not None and (type(command_contract) is not dict or
            set(command_contract) - {'expected_processor', 'expected_action_names', 'expected_consumer_namespace'}):
        raise InputError('Command contract accepts only expected_processor, expected_action_names and expected_consumer_namespace')
    if named_contract is not None and (type(named_contract) is not dict or
            set(named_contract) - {'expected_action_names', 'expected_driver_namespace', 'require_canonical_matches_driver_return', 'require_recorded_matches_driver_return'}):
        raise InputError('Named contract accepts only expected_action_names, expected_driver_namespace and strict Boolean canonical/recorded driver-return equality requirements')
    if named_contract is not None and command_contract is not None:
        raise InputError('Named and command contracts require different capture families')
    original = Path(capture).expanduser()
    if (original / 'selection.json').exists():
        if (original / 'capture.json').exists():
            raise InputError('A capture must identify exactly one runtime family')
        if named_contract is not None:
            raise InputError('Named contract requires a named-action capture')
        if command_contract is not None:
            raise InputError('Command contract requires a command-path capture')
        if dataset is not None or any(value is not None for value in
                                       (expected_names, expected_frame, expected_quantity, expected_identity)):
            raise InputError('Selection captures do not accept dataset or layout criteria')
        if max_query_gap is not None:
            from .cadence import audit_query_cadence

            return audit_query_cadence(capture, max_query_gap=max_query_gap, atol=atol, rtol=rtol)
        from .selection import audit_selection_capture

        return audit_selection_capture(capture, atol=atol, rtol=rtol)
    if max_query_gap is not None:
        raise InputError('max_query_gap requires an ACT selection capture')
    root = Path(capture).expanduser().resolve()
    manifest_path = contained(root, 'capture.json')
    manifest = read_json(manifest_path)
    if isinstance(manifest, dict) and manifest.get('kind') == 'named_action_capture':
        if command_contract is not None or dataset is not None or any(v is not None for v in
                (expected_names, expected_frame, expected_quantity, expected_identity)):
            raise InputError('Named-action capture does not accept dataset, command or layout criteria')
        from .named_actions import audit_named_actions

        return audit_named_actions(capture, **(named_contract or {}), atol=atol, rtol=rtol)
    if named_contract is not None:
        raise InputError('Named contract requires a named-action capture')
    if isinstance(manifest, dict) and manifest.get('kind') == 'command_path_capture':
        if dataset is not None or any(v is not None for v in (expected_names, expected_frame, expected_quantity, expected_identity)):
            raise InputError('Command-path capture does not accept dataset or layout criteria')
        from .command_path import audit_command_capture

        return audit_command_capture(capture, **(command_contract or {}), atol=atol, rtol=rtol)
    if command_contract is not None:
        raise InputError('Command contract requires a command-path capture')
    if isinstance(manifest, dict) and manifest.get('kind') == 'resolved_layout_capture':
        if dataset is not None:
            raise InputError('A resolved-layout snapshot does not accept a LeRobot dataset argument')
        from .layouts import audit_layout_capture

        return audit_layout_capture(str(root), expected_names=expected_names, expected_frame=expected_frame,
            expected_quantity=expected_quantity, expected_identity=expected_identity, atol=atol, rtol=rtol)
    if (any(value is not None for value in (expected_names, expected_frame, expected_quantity, expected_identity))
            or atol != 1e-6 or rtol != 1e-5):
        raise InputError('Layout acceptance parameters require a resolved-layout snapshot')
    manifest_hash = sha256(manifest_path)
    ready_path = contained(root, 'READY.json')
    ready = read_json(ready_path)
    ready_hash = sha256(ready_path)
    if type(ready.get('schema_version')) is not int or ready['schema_version'] != 1 or ready.get('manifest_sha256') != manifest_hash:
        raise InputError('Capture commit marker digest does not match the manifest')
    entries = manifest.get('files')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or type(manifest.get('has_deployment')) is not bool:
        raise InputError('Unsupported capture manifest')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 205:
        raise InputError('Capture manifest requires bounded file entries')
    listed = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
            raise InputError('Malformed capture file entry')
        path = contained(root, entry['path'])
        if str(path) in listed or not re.fullmatch('[0-9a-f]{64}', str(entry.get('sha256', ''))):
            raise InputError('Duplicate capture file or invalid digest')
        if path.suffix not in ('.json', '.safetensors') or not path.is_file():
            raise InputError('Capture entries must reference existing JSON/statistics files')
        limit = MAX_JSON_BYTES if path.suffix == '.json' else MAX_STATS_BYTES
        if path.stat().st_size > limit or sha256(path) != entry['sha256']:
            raise InputError('Capture artifact exceeds limit or digest does not match')
        listed[str(path)] = entry['sha256']
    report = audit(str(contained(root, 'checkpoint')), dataset,
                   str(contained(root, 'deployment.json')) if manifest['has_deployment'] else None)
    consumed = {a['path'] for a in report['artifacts'] if not a['role'].startswith('dataset_')}
    if consumed != set(listed):
        raise InputError('Capture manifest must cover exactly the consumed checkpoint/contract artifacts')
    if sha256(ready_path) != ready_hash or sha256(manifest_path) != manifest_hash or any(sha256(Path(path)) != digest for path, digest in listed.items()):
        raise InputError('Capture changed during verification')
    report['capture'] = {'path': str(root), 'manifest_sha256': manifest_hash, 'integrity': 'verified',
                         'recorded_engine_version': manifest.get('engine_version'), 'engine_version': __version__,
                         'versions': manifest.get('versions'), 'converters': manifest.get('converters'),
                         'scope': manifest.get('scope')}
    bindings = manifest.get('relative_bindings')
    if not isinstance(bindings, list) or len(bindings) > 100:
        raise InputError('Capture must record bounded relative processor bindings')
    report['capture']['relative_bindings'] = bindings
    saved = load_checkpoint(str(root / 'checkpoint'))
    expected_posts = {i for i, step in enumerate(saved.post)
                      if step_kind(step['name']) == 'absolute' and step['config'].get('enabled')}
    enabled_pres = {i for i, step in enumerate(saved.pre)
                    if step_kind(step['name']) == 'relative' and step['config'].get('enabled')}
    observed_posts = []
    for binding in bindings:
        if (not isinstance(binding, dict) or type(binding.get('post_step')) is not int or binding['post_step'] < 0 or
                (binding.get('pre_step') is not None and (type(binding['pre_step']) is not int or binding['pre_step'] < 0)) or
                binding.get('status') not in ('paired', 'missing', 'external', 'not_inspected')):
            raise InputError('Malformed captured relative binding')
        observed_posts.append(binding['post_step'])
        if binding['status'] == 'paired' and binding.get('pre_step') not in enabled_pres:
            raise InputError('Paired binding must identify an enabled relative preprocessor')
        if binding['status'] != 'paired':
            status = 'error' if binding['status'] == 'missing' else 'unknown'
            report['findings'].append({'code': 'capture.relative_binding', 'status': status,
                'message': 'Enabled absolute decoder binding to the captured preprocessor is ' + binding['status'] + '.',
                'evidence': [{'path': str(manifest_path), 'sha256': manifest_hash, 'pointer': '/relative_bindings',
                              'value': bindings, 'value_kind': 'source'}],
                'remedy': 'Reconnect the enabled absolute decoder to the same relative step used during preprocessing.',
                'verification': 'Object linkage alone does not establish cached-anchor freshness or action-queue timing.'})
            report['counts'][status] += 1
            if report['status'] != 'error':
                report['status'] = status
    if len(observed_posts) != len(set(observed_posts)) or set(observed_posts) != expected_posts:
        raise InputError('Capture must record every enabled absolute decoder binding exactly once')
    converters = manifest.get('converters')
    if not isinstance(converters, dict) or set(converters) != {'pre', 'post'}:
        raise InputError('Capture manifest must record both converter sets')
    custom = False
    for config in converters.values():
        if not isinstance(config, dict) or set(config) != {'to_transition', 'to_output', 'before_step_hooks', 'after_step_hooks'}:
            raise InputError('Invalid captured converters')
        if any(config[key] is not None and (not isinstance(config[key], str) or not config[key])
               for key in ('to_transition', 'to_output')):
            raise InputError('Converter identities must be nonempty strings or null')
        if any(not isinstance(config[key], list) or len(config[key]) > 100 or
               any(not isinstance(hook, str) or not hook for hook in config[key])
               for key in ('before_step_hooks', 'after_step_hooks')):
            raise InputError('Captured hooks must be bounded lists of nonempty identities')
        custom |= any(config.get(key) is not None and config[key] not in STANDARD_CONVERTERS
                      for key in ('to_transition', 'to_output'))
        custom |= bool(config.get('before_step_hooks') or config.get('after_step_hooks'))
    if custom:
        report['findings'].append({'code': 'capture.converters', 'status': 'unknown',
            'message': 'Custom converter/hook identities are captured; their implementations and effects are unverified.',
            'evidence': [{'path': str(manifest_path), 'sha256': manifest_hash, 'pointer': '/converters',
                          'value': converters, 'value_kind': 'source'}],
            'remedy': 'Validate the effective converters/hooks on recorded inputs before trusting pipeline equivalence.',
            'verification': ''})
        report['counts']['unknown'] += 1
        if report['status'] != 'error':
            report['status'] = 'unknown'
    report['limitations'].append('Runtime capture records trusted caller-supplied objects; converter implementations, driver use and declared semantics are not independently verified.')
    return report
