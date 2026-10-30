"""Verified effective-processor comparison; differences are candidate explanations."""
from __future__ import annotations

from pathlib import Path

from .artifacts import InputError, load_checkpoint, load_deployment
from .capture import audit_capture
from .compare import compare_values


def _contract(cp, capture, deployment):
    return {'policy_config': cp.config,
            **{side: {str(i): {'name': step['name'], 'config': step['config'], 'statistics': step['stats'],
                             'auxiliary_statistics': step.get('auxiliary_stats', {})}
                      for i, step in enumerate(getattr(cp, side))} for side in ('pre', 'post')},
            'converters': capture['converters'], 'relative_bindings': capture['relative_bindings'],
            'declared_deployment': deployment, 'recorded_versions': capture['versions']}


def compare_runtime_captures(reference: str, candidate: str, *, atol=1e-6, rtol=1e-5) -> dict:
    reports = [audit_capture(path, atol=atol, rtol=rtol) for path in (reference, candidate)]
    named = [report.get('kind') == 'named_action_audit' for report in reports]
    if any(named):
        if not all(named):
            raise InputError('Compare captures from the same supported runtime family')
        from .investigation import _boundary_comparison

        return _boundary_comparison(reference, candidate, atol=atol, rtol=rtol)
    commands = [report.get('kind') == 'command_path_audit' for report in reports]
    if any(commands):
        if not all(commands):
            raise InputError('Compare captures from the same supported runtime family')
        from .command_path import compare_command_captures

        return compare_command_captures(reference, candidate, atol=atol, rtol=rtol)
    selections = [report.get('kind') == 'action_selection_audit' for report in reports]
    if any(selections):
        if not all(selections):
            raise InputError('Compare captures from the same supported runtime family')
        from .selection import compare_selection_captures

        return compare_selection_captures(reference, candidate, atol=atol, rtol=rtol)
    layouts = [report.get('kind') == 'resolved_layout_audit' for report in reports]
    if any(layouts):
        if not all(layouts):
            raise InputError('Compare captures from the same supported runtime family')
        from .layouts import compare_layout_captures

        return compare_layout_captures(reference, candidate, atol=atol, rtol=rtol)
    checkpoints = [load_checkpoint(Path(report['capture']['path']) / 'checkpoint') for report in reports]
    deployments = []
    for cp, report in zip(checkpoints, reports, strict=True):
        expected = {a['path']: a['sha256'] for a in report['artifacts']}
        if any(expected.get(a.path) != a.sha256 for a in cp.artifacts):
            raise InputError('Capture changed before parameter comparison')
        path = str(Path(report['capture']['path']) / 'deployment.json')
        deployment = load_deployment(path) if path in expected else None
        if deployment and any(expected.get(a.path) != a.sha256 for a in deployment.artifacts):
            raise InputError('Captured deployment changed before comparison')
        deployments.append(deployment)
    result = compare_values(*[_contract(cp, report['capture'], dep.config if dep else None)
        for cp, report, dep in zip(checkpoints, reports, deployments, strict=True)])
    for change in result['changes']:
        tokens = change['pointer'].split('/')[1:]
        change['pointer_kind'] = 'normalized_effective_processor_contract'
        if tokens[0] == 'declared_deployment':
            change['pointer_kind'] = 'unverified_runtime_declaration'
            change['evidence'] = [{'path': a.path, 'sha256': a.sha256,
                                   'pointer': '/' + '/'.join(tokens[1:])}
                                  for dep in deployments if dep for a in dep.artifacts]
        if 3 <= len(tokens) <= 5 and tokens[0] in ('pre', 'post') and tokens[2] in ('statistics', 'auxiliary_statistics'):
            if tokens[2] == 'auxiliary_statistics':
                change['pointer_kind'] = 'normalized_auxiliary_statistic'
            side, index, category = tokens[:3]
            feature = tokens[3].replace('~1', '/').replace('~0', '~') if len(tokens) >= 4 else None
            stat = tokens[4] if len(tokens) == 5 else None
            references = []
            for cp in checkpoints:
                steps = getattr(cp, side)
                if int(index) >= len(steps) or not steps[int(index)].get('state_file'):
                    continue
                step = steps[int(index)]
                artifact = next(a for a in cp.artifacts if a.path == step['state_file'])
                statistics = step['stats'] if category == 'statistics' else step.get('auxiliary_stats', {})
                for native_feature, values in statistics.items():
                    if feature is not None and native_feature != feature:
                        continue
                    for native_stat in values:
                        if stat is None or native_stat == stat:
                            references.append({'path': artifact.path, 'sha256': artifact.sha256,
                                               'format': 'safetensors', 'tensor': native_feature + '.' + native_stat})
            change['evidence'] = references
    final = [audit_capture(path) for path in (reference, candidate)]
    if any(a['artifacts'] != b['artifacts'] or a['capture']['manifest_sha256'] != b['capture']['manifest_sha256']
           for a, b in zip(reports, final, strict=True)):
        raise InputError('Capture changed during parameter comparison')
    return {'schema_version': 1, 'status': 'pass' if result['equal'] else 'different',
            'equal': result['equal'], 'changes': result['changes'],
            'reference_audit_status': reports[0]['status'], 'candidate_audit_status': reports[1]['status'],
            'captures': [report['capture'] for report in reports],
            'artifacts': [report['artifacts'] for report in reports],
            'scope': 'Verified captured normalized configuration/statistics (including separately retained native count/q50 metadata), converter/linkage identities, recorded versions and declared deployment differences. Auxiliary statistics do not certify sample independence or calibration. Deployment semantics remain unverified declarations; tensor dtype/byte equality, model weights and transient state are outside equality scope. Changes can be intentional. Test explanations on the same saved model output; no causal or physical repair is established.'}
