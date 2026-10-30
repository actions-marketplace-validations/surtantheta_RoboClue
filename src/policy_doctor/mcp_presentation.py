"""Product-only MCP text views; the structured evidence is unchanged.

Legacy transports keep their existing representation. A human-readable text
view avoids sending a second copy of a full JSON report to an MCP client.
"""
from __future__ import annotations

import json
import re

TEXT_LIMIT = 32 * 1024


def _message(value) -> str:
    from .product_cli import _readable_message

    value = _readable_message(value)
    if not isinstance(value, str):
        return 'Inspect the cited values in structuredContent.'
    if len(value.encode()) <= 1024:
        return value
    sentences = re.split(r'(?<=[.!?])\s+', value)
    first = sentences[0]
    if len(first.encode()) <= 900:
        return first + ' Further conditions are in structuredContent.'
    return 'The complete message requires structuredContent; no sentence was cut off.'


def _parts(report: dict):
    original = report.get('report') if isinstance(report.get('report'), dict) else report
    components = report.get('components') or original.get('components') or {}
    findings = report.get('findings_shown')
    if not isinstance(findings, list):
        findings = original.get('findings') or []
    return original, components, findings


def _recap(report: dict) -> dict:
    original, components, findings = _parts(report)
    value = {'kind': report.get('kind'), 'status': report.get('status', 'unknown'),
             'scope': 'Bounded dataset artifacts only. No robot command/config was compared; this is not deployment approval.',
             'components': {name: {'status': part.get('status', 'unknown')}
                            for name, part in components.items() if isinstance(part, dict)},
             'findings_shown': [{key: item.get(key) for key in ('code', 'status', 'component') if item.get(key) is not None}
                                | {'message': _message(item.get('message') or item.get('reason'))}
                               for item in findings[:3] if isinstance(item, dict)],
             'findings_not_in_text': max(0, len(findings) - 3),
             'details': "Full scope, conditions and evidence are in structuredContent; response_format='detailed' performs a new read."}
    units = components.get('units')
    if isinstance(units, dict):
        value['components']['units'].update({key: units[key] for key in
            ('detected_units', 'compared_with', 'deployment_comparison', 'comparison_statement') if key in units})
    if isinstance(components.get('dataset_statistics'), dict):
        value['components']['dataset_statistics'].update(
            scope='Saved aggregate self-consistency only; statistics were not compared with data rows.',
            observed_row_statistics_compared=False)
    comparison = (components.get('numeric_rows') or {}).get('statistics_comparison') or original.get('statistics_comparison')
    if isinstance(comparison, dict):
        value['numeric_statistics_comparison'] = comparison
    return value


def _health_text(report: dict) -> str:
    recap = _recap(report)
    status = recap['status']
    lines = ['RoboClue dataset check: ' + str(status).upper() + '.']
    findings = recap['findings_shown']
    if findings:
        first = findings[0]
        lines.append(_message(first['message']))
    elif status == 'pass':
        lines.append('The covered metadata and row checks passed.')
    else:
        missing = [name.replace('_', ' ') for name, part in recap['components'].items() if part['status'] == 'unknown']
        lines.append('Still missing: ' + ', '.join(missing) + '.' if missing else 'Review the reported findings and coverage.')
    comparison = recap.get('numeric_statistics_comparison')
    if isinstance(comparison, dict):
        if comparison.get('status') == 'pass' and comparison.get('complete_cohort'):
            lines.append('Saved numeric statistics match the covered rows; image statistics and quantiles were not compared.')
        elif comparison.get('status') == 'error':
            lines.append('Saved numeric statistics disagree with the dataset rows. Rebuild them from the intended data before training.')
        else:
            reason = comparison.get('reason') or 'the complete supported numeric cohort was not established'
            lines.append('Numeric statistics were not compared: ' + _message(reason) + '.')
    elif 'dataset_statistics' in recap['components']:
        lines.append('Saved statistics were checked for self-consistency only; their values were not compared with data rows.')
    units = recap['components'].get('units', {})
    detected = units.get('detected_units') or {}
    if detected:
        from .deployment_answer import human_unit_mode

        values = detected.values() if isinstance(detected, dict) else [detected]
        modes = sorted({str(mode) for group in values for mode in group})
        lines.append('Recorded units: ' + '; '.join(human_unit_mode(mode) for mode in modes) + '.')
    lines.append('No robot command/config was compared; this is not deployment approval.')
    lines.append("Next: use preflight_check for a robot/config question, or request response_format='detailed' for evidence.")
    return '\n'.join(lines)


def workflow_tool_result(report: dict):
    """A plan stays a plan; keep machine requests without repeating their JSON."""
    from mcp.types import CallToolResult, TextContent

    lines = ['RoboClue can plan the next check; no assessment has run.']
    for option in report.get('can_do_now', [])[:3]:
        request = option.get('mcp_request') or {}
        name = request.get('name')
        lines.append('Next tool: ' + str(name) + ' (exact arguments in structuredContent).' if name else _message(option.get('description')))
    missing = report.get('missing_evidence') or []
    if missing:
        lines.append('Needed: ' + '; '.join(_message(item) for item in missing[:3]) + '.')
    if not report.get('can_do_now'):
        lines.append('Supply the named evidence before checking; file presence alone does not establish correctness.')
    return CallToolResult(content=[TextContent(type='text', text='\n'.join(lines))], structuredContent=report)


def _readable(report: dict, *, text_limit=None) -> str:
    if text_limit is None:
        text_limit = TEXT_LIMIT
    if report.get('kind') == 'provenance_register':
        from .register.engine import register_text

        return register_text(report)
    if report.get('image_contract'):
        from .fastwam_contract import evidence_line
        from .language_vocab import language_text

        image = report['image_contract']
        finding = next((f for f in report['findings'] if f['status'] == 'error' and f['code'] != 'fastwam.image_map'), image['findings'][0])
        return '\n'.join(['RoboClue checkpoint: ' + report['status'].upper() + '; image map ' + image['status'].upper() + '.',
                          _message(finding['message']), 'Evidence: ' + evidence_line(image),
                          'Next: ' + _message(finding.get('remedy', 'Inspect the independent findings.')),
                          'Scope: static files/source; active runtime, native execution and task success unassessed.'] +
                         ([language_text(report['language_vocabulary'])] if report.get('language_vocabulary') else []))
    if report.get('kind') == 'fastwam_image_contract':
        status = {'pass': 'CONSISTENT — image map only', 'error': 'ERROR', 'unknown': 'NOT CHECKED'}[report['status']]
        finding = report['findings'][0]
        return '\n'.join(['RoboClue FastWAM image map: ' + status + '.',
                          _message(finding['message']), 'Next: ' + _message(finding['remedy']),
                          'Scope: supplied files/source only; active runtime, model execution and task success were not checked.'])
    if report.get('kind') == 'pipeline_contract_check':
        from .pipeline_contracts import render_pipeline_contract

        return render_pipeline_contract(report).rstrip()
    original, components, findings = _parts(report)
    scope = original.get('scope')
    lines = [f"RoboClue {report.get('kind', 'assessment')}: {report.get('status', 'unknown').upper()}"]
    language = report.get('language_vocabulary') or original.get('language_vocabulary')
    if isinstance(language, dict):
        catalog = language.get('catalog', {})
        verdict = language.get('verdict')
        modes = language.get('match_modes') or sorted({row.get('match') for row in language.get('evaluations', [])})
        if verdict == 'IN_CATALOG':
            lead = ('Instruction has a lexical-only catalog match; native equivalence UNKNOWN.' if 'lexical' in modes
                    else 'Instruction matches linked catalog' +
                    ('; training revision is unpinned.' if catalog.get('saved_revision') is None else '.'))
        elif verdict == 'OUT_OF_CATALOG':
            lead = ('Instruction is outside observed catalog ' + str(catalog.get('repo_id')) + '@'
                    + str(catalog.get('resolved_revision') or 'unverified-local-snapshot') + '; WARNING.')
        else:
            lead = 'Instruction/catalog comparison is unresolved; no instruction equivalence or capability was established.'
        if report.get('status') == 'error':
            lines.append(lead)  # independent errors retain the first line
        else:
            lines = [lead, *lines]
        lines.append(str(language.get('provenance_note') or catalog.get('note') or 'Historical training coverage remains UNKNOWN.'))
        lines.append('Lexical catalog membership only; native token processing and physical capability remain unassessed.')
    if report.get('summary'):
        lines.append('This is a bounded summary; the original report exceeded the tool view or selected concise mode.')
    if isinstance(scope, str):
        lines.append('Scope: ' + scope)
    lines.append('StructuredContent retains the complete returned JSON view; text lists findings without duplicating numeric arrays.')
    if components:
        lines.append('Checks: ' + '; '.join(f"{name}={part.get('status', 'unknown')}" for name, part in components.items()
                                           if isinstance(part, dict)))
    used, omitted = len('\n'.join(lines).encode()), 0
    for item in findings:
        if not isinstance(item, dict):
            continue
        block = [f"[{str(item.get('status', 'unknown')).upper()}] {item.get('code', 'finding')}: " +
                 _message(item.get('message'))]
        if isinstance(item.get('remedy'), str) and item['remedy']:
            block.append('Next: ' + item['remedy'])
        for reference in item.get('evidence') or []:
            if not isinstance(reference, dict):
                continue
            block.append('Evidence: ' + json.dumps({key: reference[key] for key in ('path', 'pointer', 'sha256')
                                                    if key in reference}, ensure_ascii=False))
        text = '\n'.join(block)
        if used + len(text.encode()) > text_limit - 256:
            omitted += 1
            continue
        lines.append(text)
        used += len(text.encode()) + 1
    if omitted:
        lines.append(f'{omitted} finding block(s) omitted from text at its bound; structuredContent retains the returned view.')
    return '\n\n'.join(lines)


def evidence_tool_result(report: dict, *, detailed: bool):
    """Return one structured evidence object and a readable, nonduplicated text view."""
    from mcp.types import CallToolResult, TextContent

    language = isinstance(report.get('language_vocabulary'), dict)
    text = (_readable(report, text_limit=8192 if language and not detailed else TEXT_LIMIT)
            if detailed or report.get('kind') != 'dataset_health_card' else _health_text(report))
    return CallToolResult(content=[TextContent(type='text', text=text)], structuredContent=report)
