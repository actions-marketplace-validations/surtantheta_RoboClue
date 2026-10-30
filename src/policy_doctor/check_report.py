"""Small terminal view; full evidence stays available through --json."""

import hashlib
import json

from .artifacts import InputError

MAX_CHECK_SUMMARY_BYTES = 8 * 1024


def _summary_size(value: dict) -> int:
    # Also reserve whitespace for FastMCP's separate indented JSON text view.
    return max(len(json.dumps(value, allow_nan=False).encode('utf-8')),
               len(json.dumps(value, allow_nan=False, indent=2).encode('utf-8')))


def _preview(value: object, limit: int = 240) -> dict:
    text = json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    return {'text': text if len(text) <= limit else text[:limit - 3] + '...',
            'truncated': len(text) > limit, 'original_json_bytes': len(text.encode('utf-8'))}


def concise_check_summary(result: dict, *, source: str, dataset: str | None = None,
                          revision: str = 'main', dataset_revision: str = 'main',
                          scope: str = 'deployment') -> dict:
    """A bounded preview; the digest binds the full original report, not a cache."""
    encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    arguments = {'source': source, 'dataset': dataset, 'revision': revision,
                 'dataset_revision': dataset_revision, 'scope': scope, 'response_format': 'detailed'}
    unresolved = []
    remote = False
    for value, kind, key in ((source, 'model', 'revision'), (dataset, 'dataset', 'dataset_revision')):
        if value is None or not value.startswith('hf://'):
            continue
        remote = True
        pinned = next((p.get('resolved_revision') for p in result.get('provenance', [])
                       if p.get('source') == value and p.get('repo_type') == kind), None)
        if isinstance(pinned, str) and len(pinned) == 40 and all(c in '0123456789abcdef' for c in pinned):
            arguments[key] = pinned
        else:
            unresolved.append(value)
    findings = result.get('findings', [])
    unassessed = result.get('unassessed_findings', [])
    summary = {'schema_version': 1, 'kind': 'checkpoint_check_summary', 'response_format': 'concise',
        'status': result.get('status', 'unknown'), 'counts': dict(result.get('counts', {})),
        'scope': result.get('scope', ''), 'assessment_scope': result.get('assessment_scope', scope),
        'deployment_status': result.get('deployment_status', 'not_assessed'),
        'size_contract': 'At most 8192 UTF-8 bytes for JSON with ensure_ascii=True, allow_nan=False, '
                         'default separators, with no indent or indent=2. The MCP envelope contains duplicate '
                         'text/structured views and is not covered by this payload bound.',
        'detail_complete': False, 'preview': True,
        'finding_count': len(findings), 'shown_findings': 0, 'omitted_findings': len(findings), 'findings': [],
        'unassessed_findings_count': len(unassessed),
        'unassessed_findings_note': 'Unassessed findings are excluded from scoped counts, not resolved. Deployment is NOT ASSESSED.',
        'full_report': {'sha256': hashlib.sha256(encoded).hexdigest(), 'size_bytes': len(encoded),
            'canonicalization': 'JSON sorted keys; compact separators; ensure_ascii=False; allow_nan=False; UTF-8'},
        'detailed_request': {'tool': 'check_policy_metadata', 'arguments': arguments},
        'detailed_request_pin_status': 'unresolved' if unresolved else 'pinned' if remote else 'local',
        'unresolved_revisions': unresolved,
        'detailed_request_note': 'A new read, not a cached report. API receipts may differ at the same commit; compare evidence file hashes. '
                                 'Unresolved revisions are not immutable replay pins.'}
    if _summary_size(summary) > MAX_CHECK_SUMMARY_BYTES:
        raise InputError('Exact replay request cannot fit the 8 KiB concise bound; use response_format=detailed')
    ranked = sorted(findings, key=lambda f: {'error': 0, 'warning': 1, 'unknown': 2, 'pass': 3}.get(f.get('status'), 4))
    for finding in ranked:
        if finding.get('status') == 'pass' or len(summary['findings']) >= 3:
            continue
        evidence = finding.get('evidence', [])
        item = {'code': finding.get('code'), 'status': finding.get('status'),
                'message': str(finding.get('message', ''))[:240],
                'message_truncated': len(str(finding.get('message', ''))) > 240,
                'evidence': [], 'evidence_count': len(evidence), 'omitted_evidence': len(evidence), 'preview': True}
        for witness in evidence[:2]:
            selected = {key: witness[key] for key in ('path', 'sha256', 'pointer') if key in witness}
            if set(selected) != {'path', 'sha256', 'pointer'}:
                continue
            selected['value_preview'] = _preview(witness.get('value'))
            if 'source_value' in witness:
                selected['source_value_preview'] = _preview(witness['source_value'])
            if _summary_size(selected) <= 1200:
                item['evidence'].append(selected)
        item['omitted_evidence'] = len(evidence) - len(item['evidence'])
        summary['findings'].append(item)
        summary['shown_findings'] = len(summary['findings'])
        summary['omitted_findings'] = len(findings) - summary['shown_findings']
        while _summary_size(summary) > MAX_CHECK_SUMMARY_BYTES and item['evidence']:
            item['evidence'].pop()
            item['omitted_evidence'] += 1
        if _summary_size(summary) > MAX_CHECK_SUMMARY_BYTES:
            summary['findings'].pop()
            summary['shown_findings'] = len(summary['findings'])
            summary['omitted_findings'] = len(findings) - summary['shown_findings']
    return summary


def _short(value: object, limit: int = 150) -> str:
    text = json.dumps(value, ensure_ascii=True, separators=(',', ':'))
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _clip(text: str, limit: int) -> str:
    """Shorten prose at a sentence or word boundary, marking the cut; never end mid-word."""
    if len(text) <= limit:
        return text
    head = text[:limit - 4]
    sentence = head.rfind('. ')
    cut = sentence + 1 if sentence >= limit // 2 else head.rfind(' ')
    if cut < limit // 2:  # no boundary in the second half, e.g. one long token: cut hard instead of losing it
        cut = len(head)
    return head[:cut].rstrip() + ' ...'


def concise_check(result: dict) -> str:
    """Prioritize supported contradictions without hiding omitted evidence counts."""
    counts = result.get('counts', {})
    lines = ['Policy metadata check: ' + str(result.get('status', 'unknown')).upper(),
             'Scope: saved declarations only; runtime behavior and robot success are unverified.',
             'Findings: ' + ', '.join(f'{key}={counts.get(key, 0)}' for key in ('error', 'warning', 'unknown', 'pass'))]
    if result.get('assessment_scope') == 'metadata':
        lines[0] = 'Policy metadata check (metadata scope): ' + str(result.get('status', 'unknown')).upper()
        lines[1] = 'Scope: saved metadata declarations only. Deployment: NOT ASSESSED; robot success is unverified.'
        lines.append(f"Unassessed findings: {len(result.get('unassessed_findings', []))}; these were excluded from scoped counts, not resolved.")
    findings = result.get('findings', [])
    ranked = sorted(findings, key=lambda f: {'error': 0, 'warning': 1, 'unknown': 2, 'pass': 3}.get(f.get('status'), 4))
    shown = 0
    for finding in ranked:
        if finding.get('status') == 'pass' or shown >= 3:
            continue
        block = [f"{finding.get('status', 'unknown').upper()} {finding.get('code', 'finding')}: " + _clip(str(finding.get('message', '')), 420)]
        for evidence in finding.get('evidence', [])[:2]:
            block.append('  ' + _short(evidence.get('path', evidence.get('artifact', '')), 120)
                         + ' ' + _short(evidence.get('pointer', ''), 100)
                         + ' = ' + _short(evidence.get('value'), 110))
        if finding.get('remedy'):
            block.append('Next: ' + _clip(str(finding['remedy']), 300))
        candidate = '\n'.join(lines + block)
        if len(candidate.encode('utf-8')) > 3000:
            break
        lines.extend(block)
        shown += 1
    lines.append(f'Showing {shown}/{len(findings)} findings. Add --json for every finding, hash and provenance record.')
    if not shown:
        lines.append('Next: inspect the full --json report; supply effective runtime evidence to resolve unknowns.'
                     if result.get('assessment_scope') != 'metadata' else
                     'Next: inspect --json for full scoped evidence. Deployment remains NOT ASSESSED; supply a deployment contract to audit runtime declarations.')
    return '\n'.join(lines) + '\n'
