"""Small human report for portable saved-artifact checks."""
import json
import shlex


def concise_retained_check(result: dict) -> str:
    lines = [(f"Saved check: {result['retained_status']}; finding: {result.get('status', 'unknown')}"
              if 'retained_status' in result else f"Retained check: {result.get('status', 'unknown')}"),
             'Check: ' + str(result.get('check', 'unknown'))]
    if result.get('original_status') is not None:
        lines.append('Original finding: ' + str(result['original_status']))
    if 'candidate_evaluated' in result:
        lines.append('Later input evaluated: ' + str(result['candidate_evaluated']))
    if result.get('candidate_status') is not None:
        lines.append('Later finding: ' + str(result['candidate_status']))
    if 'reproduction_matches' in result:
        lines.append(f"Recorded finding reproduced: {result['reproduction_matches']}; verifier matches: {result.get('verifier_matches')}.")
    report = result.get('check_result') or {}
    if report.get('counts'):
        lines.append(' | '.join(f'{name}: {value}' for name, value in report['counts'].items()))
    for finding in report.get('findings', [])[:3]:
        lines.append(f"[{finding.get('status', 'unknown').upper()}] {finding.get('code', 'finding')}: {finding.get('message', '')}")
        for evidence in finding.get('evidence', [])[:2]:
            locator = evidence.get('pointer', evidence.get('array', ''))
            lines.append('  ' + str(evidence.get('path', '')) + '#' + str(locator))
    if report.get('first_observed_difference'):
        first = report['first_observed_difference']
        lines.append(f"First observed difference: {first.get('stream')} row {first.get('row')}, coordinate {first.get('coordinate')}.")
    if 'comparison_complete' in report:
        lines.append('Compared scope complete: ' + str(report['comparison_complete']))
    if result.get('directory'):
        lines.append('Saved: ' + json.dumps(result['directory']))
        lines.append('Reproduce: ' + shlex.join(['policy-doctor', 'reproduce-case', result['directory']]))
        lines.append('Later input: ' + shlex.join(['policy-doctor', 'check-case', result['directory'], 'LATER_NATIVE_INPUT']))
    digest = result.get('source_report_sha256', report.get('source_report_sha256'))
    if digest:
        lines.append('Report SHA256: ' + digest)
    lines.append(result.get('scope', 'Saved-artifact evidence only; physical behavior and causation remain unverified.'))
    lines.append('Use --json for evidence previews and frozen criteria; expected.json retains the full original finding.')
    return '\n'.join(lines) + '\n'
