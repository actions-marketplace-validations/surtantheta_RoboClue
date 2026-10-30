"""Small portable Markdown reports; artifact text remains escaped data."""
from __future__ import annotations

import html
import json
import re


def _text(value) -> str:
    # Values are Markdown text, never HTML attributes: retain readable quotes while
    # escaping HTML tags/entities and Markdown syntax from untrusted artifacts.
    text = html.escape(str(value), quote=False).replace('\n', ' ').replace('\r', ' ')
    for token in ('\\', '`', '|', '[', ']', '*', '_'):
        text = text.replace(token, '\\' + token)
    return text


# Characters that could end a code span, open HTML or split a table; values containing them stay escaped text.
_UNSAFE_IN_CODE = re.compile(r"[<>&|`\\\r\n]")


def _path(value) -> str:
    """Paths (and argument JSON) go in a code span so they copy exactly; Markdown escapes would change them."""
    text = str(value)
    return _text(text) if not text.strip() or _UNSAFE_IN_CODE.search(text) else f"`{text}`"


def _comparison_witness(result):
    if (result.get('comparison') or {}).get('kind') in ('native_processor_source_comparison', 'native_observation_source_comparison'):
        return None, []
    comparison = result.get('comparison') or {}
    if not isinstance(comparison, dict):
        return None, []
    first = (comparison.get('first_supported_difference') or comparison.get('first_difference')
             or comparison.get('first_layout_difference') or comparison.get('first_named_difference'))
    rows = []
    if not first:
        return None, rows
    if 'reference_body' in first:
        rows.append((f"BODY slot {first['slot']}", first['reference_body'], first['candidate_body']))
    elif 'reference_value' in first:
        rows.append((first.get('component', 'raw component'), first['reference_value'], first['candidate_value']))
    elif isinstance(first.get('reference'), dict) and isinstance(first.get('candidate'), dict):
        for name in dict.fromkeys([*first['reference'], *first['candidate']]):
            rows.append((name, first['reference'].get(name), first['candidate'].get(name)))
    elif isinstance(first.get('reference'), list) and isinstance(first.get('candidate'), list):
        for index, (left, right) in enumerate(zip(first['reference'], first['candidate'], strict=False)):
            if left != right:
                rows.append((f"{first.get('stage', 'value')}[{index}]", left, right))
    else:
        rows.append((first.get('stage', 'value'), first.get('reference'), first.get('candidate')))
    return first, rows[:8]


def _cohort_details(observed):
    lines = ['Observed evaluation comparison: ' + observed['status'] + '.']
    for role in ('reference', 'candidate'):
        summary = observed.get(role, {})
        lines.append(f"{role}: {summary.get('group_count', 0)} groups; {summary.get('total_records', 0)} observed episodes; "
                     f"Boolean true/false {summary.get('true_count', 0)}/{summary.get('false_count', 0)}; "
                     f"observed rate {summary.get('rate')}; identity inventory complete {summary.get('identity_inventory_complete', False)}.")
        artifact = observed.get('reference_intake' if role == 'reference' else 'source_intake', {}).get('artifact')
        if artifact:
            lines.append(f"Evidence: {artifact.get('path')}; SHA-256 {artifact.get('sha256')}.")
    comparison = observed.get('cohort_comparison', {})
    lines.append('Observed identities: ' + str(comparison.get('status', 'unknown')) +
                 '; selected context is checked only at the supplied pointers.')
    fields = observed.get('fields', [])
    matched = sum(field['status'] == 'pass' for field in fields)
    lines.append(f'Selected context: {matched}/{len(fields)} saved fields match; changed or missing fields follow.')
    for field in [field for field in fields if field['status'] != 'pass'][:16]:
        lines.append(field['pointer'] + ': ' + field['status'] + '; ' + field.get('reason', ''))
        for role in ('reference', 'candidate'):
            cell = field.get(role) or {}
            if 'value' in cell:
                lines.append(role + ': ' + json.dumps(cell['value'], sort_keys=True)[:512])
            if cell.get('sha256'):
                lines.append('Evidence: ' + str(cell.get('path')) + '#' + field['pointer'] + '; SHA-256 ' + cell['sha256'])
    for finding in observed.get('findings', [])[:8]:
        lines.append(f"{finding['status']}: {finding['code']}; {finding['message']}")
    lines.append('Declared counts, shards, terminal/retry completeness and native summaries: unknown. Physical pairing and causal policy improvement: unknown.')
    return lines


def _cohort_investigation_lines(result):
    lines = _cohort_details(result['audits'][0])
    if result.get('case_recipe'):
        lines.append('A comparison-selection case can retain this criterion and the supplied accepted-context reference. '
                     'Explicitly nominate that reference and choose a destination; this is not a behavioral repair. '
                     'Use the JSON case_recipe for creator arguments.')
    else:
        lines.append('No incident recipe: a healthy comparison or insufficient evidence does not establish a retained error.')
    lines.append('Use JSON for all field values, native pointers, hashes and the full bounded findings.')
    return lines


def _investigation_details(result, *, include_rows=True):
    lines = []
    for observed in result.get('audits', [])[:2]:
        contract = observed.get('source_criterion', observed.get('acceptance_contract', {}))
        if observed.get('kind') == 'native_observation_source_audit':
            lines.append('Input preservation criterion: ' + contract.get('status', 'unknown') + '; ' + contract.get('reason', ''))
            lines.append('Native processing: ' + observed.get('native_processing_status', 'unknown')
                + '; state component: ' + observed.get('native_state_component_status', 'unknown')
                + '; native numerical replay: ' + observed.get('native_replay_status', 'unknown') + '.')
            for finding in observed.get('findings', [])[:8]:
                lines.append(json.dumps(finding, sort_keys=True, allow_nan=False)[:1024])
            lines.append('Scope: declared pre-normalization API values. Physical camera identity, calibration, training membership and task competence remain unknown.')
        elif observed.get('kind') == 'native_processor_source_audit':
            lines.append('Source criterion: ' + contract.get('status', 'unknown') + '; ' + contract.get('reason', ''))
            selection = observed.get('source_selection', {})
            if selection:
                lines.append('Selected checkpoint: ' + str(selection.get('selected_checkpoint', 'unknown'))
                    + '; captured consumer source argument: ' + str(selection.get('consumer_argument', 'unknown')) + '.')
            lines.append('Native preprocessing: ' + observed.get('native_processing_status', 'unknown')
                + '; captured same-input output parity: ' + observed.get('captured_output_parity_status', 'unknown')
                + '; native numerical replay: ' + observed.get('native_replay_status', 'unknown') + '.')
            for side in ('consumer', 'reference'):
                keys = sorted({key for item in observed.get('normalizer_input_statistics', {}).get(side, [])
                               for key in item.get('observed_input_stat_keys', [])})
                lines.append(side + ' observed input keys with matching nonidentity statistics: '
                    + (', '.join(keys) if keys else 'none evidenced') + '.')
        else:
            lines.append(f"Observed {observed.get('kind', 'native contract')}: {observed['status']}; explicit criterion {contract.get('status', 'not supplied')}.")
        if observed.get('kind') == 'evaluation_cohort_audit':
            lines.extend(_cohort_details(observed))
        elif observed.get('kind') == 'native_run_context_audit':
            lines.append('Recorded comparison eligibility: ' + observed['status'] + '; selected settings only, effective runtime and asset identity remain unknown.')
            for field in observed.get('fields', [])[:8]:
                lines.append(field['pointer'] + ': ' + field['status'] + '; ' + field.get('reason', ''))
                for role in ('reference', 'candidate'):
                    cell = field.get(role, {})
                    if 'value' in cell:
                        encoded = json.dumps(cell['value'], sort_keys=True)
                        lines.append(role + ': ' + encoded[:512] + (' … (preview)' if len(encoded) > 512 else ''))
                    if cell.get('sha256'):
                        lines.append('Evidence: ' + str(cell.get('path')) + '#' + field['pointer'] + '; SHA-256 ' + cell['sha256'])
        elif observed.get('kind') == 'named_action_audit':
            recording = observed.get('recording', {})
            lines.append(f"Named software boundary: binding {observed['binding_status']}; numeric replay {observed['numeric_replay_status']}; driver changes {observed['driver_change_events']}.")
            lines.append(f"Native recording: buffer {recording.get('buffer_binding_status', 'unknown')}; persistence {recording.get('persistence_status', 'unknown')}; verified rows {recording.get('rows_checked', 0)}.")
            if contract.get('require_recorded_matches_driver_return') is True:
                lines.append('Explicit requirement: persisted native action values must match the observed driver return.')
            if contract.get('require_canonical_matches_driver_return') is True:
                lines.append('Explicit requirement: canonical input must match the observed driver return.')
            if recording.get('status') == 'pass':
                lines.append('The writer consumed the canonical native return; equal numbers do not make its source the driver-return object.')
            if recording.get('reason'):
                lines.append('Recording evidence needed: ' + recording['reason'])
            lines.append('Legal limiting is an observation unless your explicit requirement prohibits it. Physical execution and earlier policy roles remain unknown.')
        elif observed.get('kind') == 'resolved_layout_audit':
            lines.append(f"Raw term binding: {observed['raw_binding']['status']}; processed stage: {observed['processed_binding']['status']}. Physical sampling remains unknown.")
            if contract.get('expected_names') is not None:
                names = contract['expected_names']
                lines.append('Required BODY order: ' + ' → '.join(names[:8]) + (f' … ({len(names)} names)' if len(names) > 8 else ''))
                lines.append('Observed BODY order: ' + ' → '.join((observed['layout'].get('body_names') or [])[:8]))
            for key in ('expected_frame', 'expected_quantity', 'expected_identity'):
                if contract.get(key) is not None:
                    lines.append(key.replace('_', ' ') + ': ' + str(contract[key]))
    comparison = result.get('comparison')
    if comparison and comparison.get('kind') in ('native_processor_source_comparison', 'native_observation_source_comparison'):
        lines.append('Reference source criterion: ' + comparison.get('reference_audit_status', 'unknown')
            + '; same original correction input: ' + str(comparison.get('same_correction_input', False)) + '.')
        if comparison['kind'] == 'native_observation_source_comparison':
            lines.append('The retained rule checks declared state/camera API values and model metadata identity. Physical effects remain unknown.')
        else:
            lines.append('The source rule applies to declared restoration intent. Policy competence and physical effects remain unknown.')
    elif comparison:
        lines.append('Reference comparison: ' + comparison.get('alignment', comparison.get('scope', 'supported native family only')))
        if 'reference_criterion_status' in comparison:
            lines.append('Reference explicit criterion: ' + comparison['reference_criterion_status'])
        first, rows = _comparison_witness(result)
        if first:
            stage = first.get('stage', 'BODY order' if 'reference_body' in first else 'named raw value')
            lines.append(f"First supported difference: {stage}" + (f" at native send ordinal {first['tick_ordinal']}" if 'tick_ordinal' in first else ''))
            if include_rows:
                lines.append('Component | Reference | Candidate')
                lines.extend(f'{name} | {left} | {right}' for name, left, right in rows)
            if first.get('omitted_changed_channel_count'):
                lines.append(f"Additional changed channels omitted: {first['omitted_changed_channel_count']}; exact total {first['changed_channel_count']}.")
            for evidence in first.get('evidence', [])[:2]:
                location = evidence.get('pointer', 'row=' + str(evidence.get('row')) + ', column=' + str(evidence.get('column')))
                if 'array' in evidence:
                    location = evidence['array'] + str(evidence.get('index'))
                lines.append(f"Evidence: {evidence['path']}#{location}; SHA-256 {evidence['sha256']}")
        elif comparison.get('comparison_supported') is False:
            lines.append('Comparison unknown: ' + comparison.get('reason', 'native evidence is insufficient'))
        else:
            lines.append('No difference reported over the supported compared roles; missing roles and unmatched calls remain unknown.')
        lines.append('Observed differences do not by themselves establish a fault or physical cause.')
    if result.get('case_recipe'):
        if result['case_recipe'].get('retained_context'):
            lines.append('Case reuse also retains these known corrected-source context invariants: ' + str(result['case_recipe']['retained_context']))
        if result.get('applied_requirements', {}).get('kind') == 'native-run-context':
            lines.append('Create a portable recorded-comparison case using this requirement and supplied reference. Numeric extraction and image omissions will be recorded; no case was written.')
        else:
            lines.append('Verified supplied correction is ready for the existing case creator. Choose a new destination and reuse this requirements file; no case was written.')
    return lines


def markdown_report(result: dict) -> str:
    if result.get('kind') in ('video_reference_recovery', 'video_reference_recovery_reproduction',
                              'video_reference_recovery_candidate'):
        lines = ['# Video reference recovery', '', f"Report status: **{_text(result['status'])}**", '',
                 f"Original interval status: **{_text(result['original_status'])}**", '',
                 f"Repair verified: **{_text(result['repair_verified'])}**", '',
                 f"Origin verified: **{_text(result.get('origin_verified', False))}**", '',
                 _text(result.get('reason', '')), '', 'Episode: ' + _text(result['episode_index'])]
        for key, label in (('candidate_episode_index', 'Candidate episode'),
                           ('reproduction_matches', 'Report reproduced'),
                           ('verifier_matches', 'Verifier matches'),
                           ('candidate_evaluated', 'Candidate evaluated'),
                           ('distinct_episode', 'Distinct episode'),
                           ('fixed_criterion_matches', 'Frozen criterion matches'),
                           ('shared_origin_bytes_consistent', 'Shared origin bytes consistent'),
                           ('recovered_intervals_disjoint', 'Recovered intervals disjoint'),
                           ('original_clips_distinct', 'Original clips distinct')):
            if key in result:
                lines.append(label + ': ' + _text(result[key]))
        if result.get('case_directory'):
            lines.extend(['', 'Saved case: ' + _path(result['case_directory'])])
        if result.get('patch'):
            lines.extend(['', '## Proposed metadata changes', '',
                          '| Path | Row | Column | Old | New |', '| --- | ---: | --- | --- | --- |'])
            for change in result['patch']:
                lines.append('| ' + ' | '.join(_text(change[key]) for key in
                             ('path', 'row', 'column', 'old', 'new')) + ' |')
        derived = result.get('derived_metadata')
        if isinstance(derived, dict):
            lines.extend(['', 'Derived metadata copy: ' + _path(derived.get('path')) + ' replaces '
                          + _path(derived.get('dataset_path')) + ' (sha256 ' + _text(derived.get('sha256')) + ')'])
        if result.get('overlay_scope'):
            lines.extend(['', _text(result['overlay_scope'])])
        if isinstance(result.get('validation_scope'), dict):
            lines.extend(['', '- Replay: ' + _text(result['validation_scope'].get('replay', '')),
                          '- Native consumer validation: ' + _text(result['validation_scope'].get('native_consumer', ''))])
        for key, label in (('original_reference', 'Original reference'),
                           ('recovered_reference', 'Recovered reference'),
                           ('origin_conflicts', 'Origin conflicts')):
            if result.get(key):
                lines.extend(['', '## ' + label, '', _text(json.dumps(result[key], sort_keys=True))])
        evidence = result.get('evidence', {})
        if evidence:
            lines.extend(['', '## Evidence', ''])
            for key in ('sample_count', 'all_samples_match', 'codec_match', 'timestamps_match',
                        'original_clip_path', 'original_clip_sha256', 'original_metadata_path',
                        'original_metadata_sha256'):
                if key in evidence:
                    lines.append('- ' + _text(key) + ': ' + _text(evidence[key]))
        lines.extend(['', '## Limits', '', _text(result.get('verification_scope', '')),
                      'Offline replay checks internal consistency; it does not authenticate origin or establish physical execution.',
                      _text(result.get('scope', ''))])
        lines.extend('- ' + _text(reason) for reason in result.get('unknowns', []))
        return '\n'.join([*lines, ''])
    if (result.get('applied_requirements') or {}).get('kind') == 'evaluation-cohort':
        return '\n'.join(['# Observed evaluation comparison', ''] +
                         [_text(line) + '\n' for line in _cohort_investigation_lines(result)]) + '\n'
    if result.get('kind') == 'evaluation_cohort_audit':
        return '\n'.join(['# Observed evaluation comparison', ''] + [_text(line) for line in _cohort_details(result)]) + '\n'
    if result.get('kind') == 'named_action_audit':
        lines = ['# Native named-action boundary', '', f"Status: **{_text(result['status'])}**", '']
        lines.extend(_text(line) for line in _investigation_details({'audits': [result]}))
        change = result.get('first_driver_change')
        if change:
            lines.extend(['', '## First observed driver change', '', '| Component | Driver request | Driver return |', '| --- | --- | --- |'])
            lines.extend('| ' + ' | '.join(_text(value) for value in (row['name'], row['driver_request'], row['driver_return'])) + ' |'
                         for row in change['channels'][:8])
            lines.append(f"Changed channels: {_text(change['changed_channel_count'])}; preview truncated: {_text(change['channels_truncated'])}.")
        recorded = result.get('recording', {}).get('first_difference')
        if recorded:
            lines.extend(['', '## First recorded/driver difference', '', '| Component | Persisted row | Driver return |', '| --- | --- | --- |'])
            names = [name for name in recorded['recorded'] if recorded['recorded'][name] != recorded['driver_return'].get(name)]
            lines.extend('| ' + ' | '.join(_text(value) for value in (name, recorded['recorded'][name], recorded['driver_return'].get(name))) + ' |'
                         for name in names[:8])
        for violation in result['acceptance_contract'].get('violations', [])[:4]:
            lines.append('Explicit requirement differs: ' + _text(violation['field']) + '.')
        lines.extend(['', '## Original evidence', ''])
        for artifact in result.get('artifacts', []):
            lines.append('- ' + _path(artifact['path']) + '; SHA-256 ' + _text(artifact['sha256']))
        lines.extend(['', _text(result.get('scope', '')), ''])
        return '\n'.join(lines)
    if result.get('kind') == 'run_boundary_comparison':
        wrapped = {'comparison': result}
        lines = ['# Observed native boundary comparison', '', *[_text(line) for line in _investigation_details(wrapped, include_rows=False)]]
        _, rows = _comparison_witness(wrapped)
        if rows:
            lines.extend(['', '| Component | Reference | Candidate |', '| --- | --- | --- |'])
            lines.extend('| ' + ' | '.join(_text(value) for value in row) + ' |' for row in rows)
        return '\n'.join([*lines, '', _text(result.get('scope', '')), ''])
    if result.get('kind') in ('command_path_audit', 'command_path_comparison'):
        lines = ['# Observed software command path', '', f"Status: **{_text(result['status'])}**", '',
                 _text(result.get('scope', '')), '']
        if result['kind'] == 'command_path_audit':
            lines.extend([f"Binding: {_text(result['binding_status'])}; numeric replay: {_text(result['numeric_replay_status'])}; explicit contract: {_text(result['acceptance_contract']['status'])}.",
                          '', '## Effective processor and explicit criteria', '',
                          'Observed processor: ' + _text(json.dumps(result.get('observed_processor'), sort_keys=True)),
                          'Acceptance contract: ' + _text(json.dumps(result['acceptance_contract'], sort_keys=True)),
                          'Declared tensor order: ' + _text(json.dumps(result['action_names'])),
                          'Software consumer namespace: ' + _text(result['consumer_namespace']), '',
                          '## Observed stages', '', '| Tick | Outcome | Binding | Numeric replay |', '| --- | --- | --- | --- |'])
            for tick in result.get('ticks', [])[:64]:
                lines.append(f"| {_text(tick['tick_id'])} | {_text(tick['outcome'])} | {_text(tick['binding_status'])} | {_text(tick['numeric_replay_status'])} |")
            lines.extend(['', '## Evidence', ''])
            for artifact in result.get('artifacts', []):
                lines.append('- ' + _path(artifact['path']) + '; SHA-256 ' + _text(artifact['sha256']))
        else:
            lines.extend([f"Comparison supported: {_text(result.get('comparison_supported'))}; equal over compared ticks: {_text(result.get('equal_over_compared_ticks'))}.", '',
                          'First differing stage: ' + _text(json.dumps(result.get('first_difference'), sort_keys=True)), '',
                          'Context differences: ' + _text(json.dumps(result.get('context_differences', []), sort_keys=True)), '',
                          'A lawful recipe or consumer-return difference is not automatically a fault.'])
        lines.extend(['', 'Declared names describe tensor order, not an executed named-joint mapping.',
                      'Physical execution: unknown. Recorded actions: unknown. Historical query lineage: unknown.', ''])
        return '\n'.join(lines)
    if result.get('kind') == 'query_cadence_audit':
        lines = ['# Observed native query cadence', '', f"Status: **{_text(result['status'])}**", '',
                 _text(result['scope']), '', '## Explicit consumer requirement', '',
                 _text(json.dumps(result['consumer_requirement'], sort_keys=True)), '',
                 '## Observed criterion', '', _text(json.dumps(result['cadence_criterion'], sort_keys=True)), '',
                 '## Evidence', '', _text(json.dumps(result['artifacts'], sort_keys=True)), '',
                 'Query lineage and downstream association remain separate from this bounded cadence check.', '']
        return '\n'.join(lines)
    if result.get('kind') in ('action_selection_audit', 'action_selection_comparison'):
        lines = ['# Saved action selection', '', f"Status: **{_text(result['status'])}**", '', _text(result['scope']), '']
        if result['kind'] == 'action_selection_audit':
            lines.extend(['| Event | Type | Numeric replay | Query origin |', '| --- | --- | --- | --- |'])
            for event in result['events']:
                lines.append(f"| {event['event_id']} | {_text(event['type'])} | {_text(event['numeric_replay']['status'])} | {_text(event['origin']['status'])} |")
            lines.extend(['', '## Selection origins', ''])
            for event in result['events']:
                if event['type'] == 'select':
                    lines.append('- Event ' + str(event['event_id']) + ': ' + _text(json.dumps(event.get('arithmetic_contributors', event['origin']), sort_keys=True)))
                    if event.get('local_binding'):
                        lines.append('  Local binding: ' + _text(json.dumps(event['local_binding'], sort_keys=True)))
            lines.extend(['', '## Evidence', ''])
            for artifact in result['artifacts']:
                lines.append('- ' + _path(artifact['path']) + '; SHA-256 ' + _text(artifact['sha256']))
        else:
            lines.extend([_text(result['alignment_basis']), '', 'First selected difference: ' + _text(json.dumps(result['first_selected_difference'], sort_keys=True)),
                          '', 'A lawful selection difference is not automatically a fault.'])
        lines.extend(['', 'Downstream decoder/issued/returned/recorded association: unknown.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') in ('resolved_layout_audit', 'resolved_layout_comparison'):
        lines = ['# Resolved raw-term layout', '', f"Status: **{_text(result['status'])}**", '',
                 _text(result.get('scope', '')), '']
        if result['kind'] == 'resolved_layout_audit':
            lines.extend(['## Resolved and expected identities', '',
                          'Resolved BODY order: ' + _text(json.dumps(result['layout'].get('body_names'))),
                          'Retained acceptance contract: ' + _text(json.dumps(result['acceptance_contract'], sort_keys=True)),
                          '', '## Saved-buffer binding', '', _text(json.dumps(result['raw_binding'], sort_keys=True)),
                          '', 'Processed stage: ' + _text(json.dumps(result['processed_binding'], sort_keys=True)),
                          '', '## Findings', ''])
            for finding in result.get('findings', []):
                lines.append('- ' + _text(json.dumps(finding, sort_keys=True)))
        else:
            lines.extend(['| Check | Result |', '| --- | --- |',
                          f"| Ordered layout equal | {_text(result.get('ordered_layout_equal'))} |",
                          f"| Raw values equal | {_text(result.get('raw_values_equal'))} |",
                          f"| Named values equal | {_text(result.get('named_values_equal'))} |", '',
                          'A valid permutation can preserve named values while changing raw order.',
                          'Matching raw numbers do not establish matching named policy slots.', '',
                          'First layout difference: ' + _text(json.dumps(result.get('first_layout_difference'))),
                          'First named value difference: ' + _text(json.dumps(result.get('first_named_difference')))])
        lines.extend(['', '## Evidence', '', _text(json.dumps(result.get('artifacts'), sort_keys=True)), '',
                      'Saved-source consistency is separate from authenticated live origin. Physical sample association and policy concatenation remain unverified.', ''])
        return '\n'.join(lines)
    if result.get('kind') == 'incident_candidate_check':
        if (result.get('candidate') or {}).get('kind') == 'evaluation_cohort_audit':
            return '\n'.join(['# Retained evaluation comparison', '', f"Status: **{_text(result['status'])}**", '',
                f"Case verified: {_text(result['case_verified'])}.", ''] +
                [_text(line) for line in _cohort_details(result['candidate'])]) + '\n'
        lines = ['# Retained incident regression', '', f"Status: **{_text(result['status'])}**", '',
                 f"Case verified: {_text(result['case_verified'])}; candidate evaluated: {_text(result['candidate_evaluated'])}.",
                 '', _text(result.get('scope', '')), '', '## Frozen check parameters', '',
                 _text(json.dumps(result.get('parameters'), sort_keys=True)), '', '## Later native artifact', '',
                 _text(json.dumps(result.get('candidate'), sort_keys=True)), '',
                 'The check applies the retained criteria to the supplied artifact. Its numerical result does not certify physical behavior.', '']
        return '\n'.join(lines)
    if result.get('kind') in ('exported_incident_case', 'incident_case_reproduction'):
        lines = ['# Verified native incident case', '', f"Case status: **{_text(result['status'])}**", '',
                 _text(result.get('scope', '')), '', '| Check | Recorded result |', '| --- | --- |']
        if result.get('retained_context'):
            lines.extend(['Additional retained corrected-source context: ' + _text(str(result['retained_context'])), ''])
        checks = result.get('checks', {})
        for name in ('before', 'after'):
            lines.append(f"| {_text(name)} | {_text(checks.get(name, {}).get('status', 'unknown'))} |")
        for control in checks.get('controls', []):
            lines.append(f"| Control: {_text(control['name'])} | {_text(control['check']['status'])}; expected {_text(control['expected_status'])} |")
        if result.get('directory'):
            lines.extend(['', 'Saved case: ' + _path(result['directory']), 'Rerun using reproduce-case with that directory.'])
        if 'reproduction_matches' in result:
            lines.extend(['', f"Recorded findings reproduced: {_text(result['reproduction_matches'])}.",
                          f"Verifier matches: {_text(result['verifier_matches'])}.",
                          f"Lifecycle verified: {_text(result.get('lifecycle_verified'))}."])
        lines.extend(['', '## Correction evidence', '', _text(json.dumps(result.get('change_verification'), sort_keys=True)),
                      '', '## Native findings', ''])
        if checks.get('before', {}).get('kind') == 'evaluation_cohort_audit':
            for role in ('before', 'after'):
                lines.extend([role + ':'] + [_text(line) for line in _cohort_details(checks[role])] + [''])
        else:
            lines.append(_text(json.dumps(checks, sort_keys=True)))
        lines.extend(['', 'Case reproduction verifies the declared numeric checks. Physical execution and policy outcomes need separate evidence.', ''])
        return '\n'.join(lines)
    if result.get('kind') == 'episode_window_audit':
        lines = ['# Native episode-window investigation', '', f"Status: **{_text(result['status'])}**", '',
                 _text(result['scope']), '', '## Metadata findings', '']
        for finding in result['findings']:
            details = [f"{_text(finding['status'])}: {_text(finding['code'])}"]
            if finding.get('episode_index') is not None:
                details.append('episode ' + _text(finding['episode_index']))
            if finding.get('field', finding.get('column')) is not None:
                details.append(_text(finding.get('field', finding.get('column'))))
            if 'declared' in finding and 'observed' in finding:
                details.extend(['declared ' + _text(finding['declared']), 'actual ' + _text(finding['observed'])])
            elif 'observed' in finding and 'expected' in finding:
                details.extend(['declared ' + _text(finding['observed']), 'actual ' + _text(finding['expected'])])
            lines.append('- ' + '; '.join(details) + '.')
            if finding.get('message'):
                lines.append('  ' + _text(finding['message']))
            for evidence in finding.get('evidence', [])[:2]:
                lines.append('  - Native source: ' + _text(json.dumps(evidence, sort_keys=True)))
        lines.extend(['', 'Metadata-only correction candidates: ' + _text(json.dumps(result.get('metadata_patches', []), sort_keys=True)),
                      ''])
        if result.get('query') is not None:
            lines.extend(['## Selected window', '', _text(json.dumps(result['query'], sort_keys=True)), ''])
            if result.get('metadata_patches') and result['query'].get('corrected_what_if') is not None:
                lines.extend(['The corrected window is a numerical what-if; rerun against a saved corrected copy.', ''])
        lines.extend(['Video alignment, policy competence and physical repair are outside this check.',
                      '', '## Coverage', '', _text(json.dumps(result['coverage'], sort_keys=True)), '', '## Evidence', ''])
        for artifact in result['artifacts']:
            lines.append(f"- {_path(artifact['path'])}; SHA-256 {_text(artifact['sha256'])}.")
        return '\n'.join(lines) + '\n'
    if result.get('kind') == 'generated_trajectory_demo':
        lines = ['# Same waypoints, different sampled path', '', _text(result['scope']), '',
                 _text(result['summary']), '',
                 '| Check | Result |', '| --- | --- |',
                 f"| Candidate position at 0.75 seconds (linear reference: 0.75) | {_text(result['candidate_position_at_750ms'][0])} |",
                 f"| Supplied fixture bounds | {_text(result['candidate_bounds_status'])} |",
                 f"| Corrected sampled reference | {_text(result['corrected_status'])} |",
                 f"| Valid joint reorder | {_text(result['valid_reordering'])} |",
                 f"| Missing initial state | {_text(result['missing_initial_state'])} |", '',
                 'Demo creation succeeded. The candidate intentionally violates supplied fixture bounds.',
                 'Execution status: **unknown**. Physical cause: **unknown**.', '', '## Check the evidence', '']
        for name, path in result['report_paths'].items():
            lines.append(f"- {_text(name)}: {_path(path)}.")
        lines.extend(['', 'Reports retain input hashes, native waypoint fields and the pinned interpolation source.',
                      '', '## Reproduce from this folder', '',
                      '```sh',
                      'policy-doctor compare-trajectories reference.json candidate.json --times 0 0.25 0.5 0.75 1 --markdown',
                      'policy-doctor compare-trajectories reference.json corrected.json --times 0 0.25 0.5 0.75 1 --markdown',
                      'policy-doctor sample-trajectory missing_initial_state.json --times 0 1 1.5 --markdown',
                      '```', '', 'Expected exits: 1 for the differing candidate, 0 for the corrected reference, 3 for missing initial state.',
                      'Edit a copy of a message and compare it at explicit times using compare-trajectories.',
                      'Reference equality is numerical; this fixture establishes no physical repair.', ''])
        return '\n'.join(lines)
    if result.get('kind') in ('joint_trajectory_samples', 'joint_trajectory_comparison'):
        comparison = result['kind'] == 'joint_trajectory_comparison'
        lines = ['# ROS trajectory reference investigation', '', f"Status: **{_text(result['status'])}**", '',
                 _text(result['scope']), '', 'Execution status: **unknown**. Physical cause: **unknown**.', '']
        if comparison:
            lines.extend([f"Same waypoint positions and times: {_text(result['same_waypoint_positions_and_times'])}.",
                          f"Joint alignment: {_text(result['joint_alignment'])}.",
                          f"Comparison complete: {_text(result['comparison_complete'])}.",
                          f"Reference bounds/sampling status: {_text(result['reference_sampling_status'])}; candidate: {_text(result['candidate_sampling_status'])}.",
                          'First position discrepancy: ' + _text(json.dumps(result['first_position_difference'])), ''])
            lines.extend(['| Differing sampled field | First time (ns) | Reference values | Candidate values |',
                          '| --- | ---: | --- | --- |'])
            for field, difference in result['different_sample_fields'].items():
                lines.append(f"| {_text(field)} | {_text(difference['time_from_start_ns'])} | {_text(difference['reference_values'])} | {_text(difference['candidate_values'])} |")
            lines.append('')
        reports = [result['reference'], result['candidate']] if comparison else [result]
        for index, report in enumerate(reports):
            lines.extend([f"## {'Reference' if index == 0 else 'Candidate'} samples", '',
                'Joints in native order: ' + _text(report['joint_names']),
                'Selected interpolation: ' + _text(report['interpolation_method']), '',
                '| Time from start (ns) | Rule | Positions | Velocities | Accelerations | Effort |', '| ---: | --- | --- | --- | --- | --- |'])
            for sample in report['samples']:
                lines.append(f"| {_text(sample['time_from_start_ns'])} | {_text(sample.get('interpolation', sample['status']))} | {_text(sample.get('positions', sample.get('reason')))} | {_text(sample.get('velocities', 'unknown'))} | {_text(sample.get('accelerations', 'unknown'))} | {_text(sample.get('effort', 'unknown'))} |")
            lines.extend(['', 'Sampled bounds violations: ' + _text(json.dumps(report['sampled_limit_violations'])),
                          '', 'Waypoint-envelope excursion hints: ' + _text(report['excursion_count']),
                          _text(json.dumps(report['waypoint_envelope_excursions'])), _text(report['excursion_scope']),
                          '', _text(report['limits_scope']), '', _text(report['header_treatment']),
                          _text(report['sample_time_treatment']), '',
                          'Unsupported: ' + ', '.join(_text(v) for v in report['unsupported']), ''])
        if result.get('case_directory'):
            lines.extend(['Saved sampling case: ' + _path(result['case_directory']),
                          'Use reproduce-case or MCP reproduce_trajectory_case with that directory.', ''])
        lines.extend(['## Evidence', ''])
        for artifact in result['artifacts']:
            lines.append(f"- {_path(artifact['path'])}; SHA-256 {_text(artifact['sha256'])}.")
        lines.extend(['', 'Pinned upstream sampling rule: ' + _text(result['upstream_reference']['url']),
                      'Upstream source SHA-256: ' + _text(result['upstream_reference']['sha256']),
                      '', 'Samples are derived values from native waypoint fields. No full-controller or physical replay is established.', ''])
        return '\n'.join(lines) + '\n'
    if 'captures' in result and 'changes' in result:
        lines = ['# Captured runtime comparison', '', f"Status: **{_text(result['status'])}**", '', _text(result['scope']), '',
                 '| Effective contract pointer | Reference | Candidate |', '| --- | --- | --- |']
        for change in result['changes']:
            lines.append(f"| {_text(change['pointer'])} | {_text(change.get('before'))} | {_text(change.get('after'))} |")
        lines.extend(['', '## Evidence', ''])
        for artifacts in result['artifacts']:
            for artifact in artifacts:
                lines.append(f"- {_path(artifact['path'])}; SHA-256 {_text(artifact['sha256'])}.")
        return '\n'.join(lines) + '\n'
    rollout = 'first_observed_difference' in result
    lines = ['# Recorded behavior comparison' if rollout else '# Robot run investigation', '',
             f"Status: **{_text(result['status'])}**", '', _text(result.get('scope', '')), '']
    if result.get('summary'):
        lines.extend([_text(result['summary']), ''])
    if rollout:
        first = result['first_observed_difference']
        selected = any(a.get('stream_selection') for a in result['artifacts'])
        coverage = 'Selected-stream comparison complete' if selected else 'Comparison complete'
        lines.extend(['## First observed difference', '',
                      _text(f"{first['stream']}: row {first['row']}, coordinate {first['coordinate']}, sample ID {first['sample_id']}.")
                      if first else 'No numeric difference observed in the compared rows.', '',
                      f"{coverage}: {_text(result['comparison_complete'])}. Causal conclusion: **unknown**.", '',
                      f"Alignment: {_text(result['alignment']['reason'])} Sample phase: {_text(result['alignment']['sample_phase'])}.", '',
                      '| Native stream | Role | First differing row | Values changed | Rows compared |',
                      '| --- | --- | ---: | ---: | ---: |'])
        for stream in result['streams']:
            point = stream.get('first_difference')
            lines.append(f"| {_text(stream['key'])} | {_text(stream.get('role', 'numeric signal'))} | {_text(point['row'] if point else stream['status'])} | {_text(stream.get('different_values', 'unknown'))} | {_text(stream['rows_compared'])} |")
        lines.extend(['', _text(result['bookkeeping_scope']), f"Bookkeeping difference: {_text(result['bookkeeping_difference'])}."])
        lines.extend(['', '## Coverage and tolerances', '', _text(json.dumps(result['tolerances'], sort_keys=True)),
                      '', 'Integer/bool signals compare exactly. Floating signals use the displayed tolerances.',
                      '', 'Unmatched streams: ' + _text(json.dumps(result['unmatched_streams'], sort_keys=True))])
        for artifact in result['artifacts']:
            lines.append(f"- {_path(artifact['path'])}: excluded fields {_text(artifact.get('excluded_streams', []))}.")
            selection = artifact.get('stream_selection')
            if selection:
                lines.extend(['', 'Selected native NPZ streams: ' + _text(json.dumps(selection['selected'])),
                              'Excluded arrays: ' + _text(json.dumps(selection['excluded'])),
                              _text(selection['scope']),
                              'Header inventory: ' + _text(json.dumps(selection['inventory'], sort_keys=True))])
        lines.extend(['', '## Evidence', ''])
        for artifact in result['artifacts']:
            lines.extend([f"- {_path(artifact['path'])}; episode {_text(artifact.get('episode', 'not selected'))}; SHA-256 {_text(artifact['sha256'])}."])
        if first:
            lines.extend(['', 'Selected values at the first differing coordinate:'])
            for evidence in first['evidence']:
                lines.append(f"- {_path(evidence['path'])}: {_text(evidence['array'])}{_text(evidence['index'])} = {_text(evidence['value'])}.")
        lines.extend(['', '## Still unknown', '', ', '.join(_text(v) for v in result['unverified_state']) + '.',
                      '', _text(result['coverage_scope']), ''])
    else:
        if 'inventory' in result:
            lines.extend(['## Observed native roles and explicit comparison', ''])
            lines.extend(_text(line) + '\n' for line in _investigation_details(result, include_rows=False))
            _, rows = _comparison_witness(result)
            if rows:
                lines.extend(['', '| Component | Reference | Candidate |', '| --- | --- | --- |'])
                lines.extend('| ' + ' | '.join(_text(value) for value in row) + ' |' for row in rows)
        lines.extend(['## Findings', ''])
        for finding in result['findings']:
            lines.append(f"- **{_text(finding['status'])}: {_text(finding['code'])}** — {_text(finding['message'])}")
            for evidence in finding.get('evidence', [])[:2]:
                lines.append(f"  - Source: {_path(evidence['path'])} #{_text(evidence.get('pointer', 'row=' + str(evidence.get('row')) + ', column=' + str(evidence.get('column'))))}; SHA-256 {_text(evidence['sha256'])}.")
        if not result['findings']:
            lines.append('No non-passing findings from the audits actually performed. Uninspected artifacts remain unverified.')
        lines.extend(['', '## Intake and evidence failures', ''])
        for problem in result.get('intake_problems', []):
            lines.append(f"- {_path(problem['path'])}: {_text(problem['reason'])}")
        if not result.get('intake_problems'):
            lines.append('No intake failures reported within the inspected scope.')
        lines.extend(['', '## Discovered candidates', ''])
        for kind, items in result.get('inventory', {}).items():
            lines.append(f"- {_text(kind)}: {_text(len(items))}")
            for item in items[:8]:
                lines.append('  - ' + _path(item['path'] if isinstance(item, dict) else item))
                if isinstance(item, dict) and item.get('needs_selection_or_adapter'):
                    lines.append('    ' + _text(item['needs_selection_or_adapter']))
        coverage = result.get('coverage') or {}
        for key, title in (('tabular_files_unrecognized', 'Tabular files not recognized as rollout recordings'),
                           ('tabular_files_not_inspected', 'Tabular files not inspected')):
            if coverage.get(key + '_count'):
                lines.extend(['', f"## {title}", '', f"{_text(coverage[key + '_count'])} file(s); first "
                              f"{_text(len(coverage.get(key, [])))} listed.", ''])
                for item in coverage.get(key, []):
                    lines.append(f"- {_path(item['path'])}: {_text(item['reason'])}")
        lines.extend(['', '## Next actions', ''])
        for action in result['next_actions']:
            lines.append(f"- {_text(action['tool'])}: {_text(action['purpose'])}")
            if action['arguments'] is not None:
                lines.append('  Arguments: ' + _path(json.dumps(action['arguments'], sort_keys=True)))
        lines.extend(['', '## Coverage', '', _text(json.dumps(result['coverage'], sort_keys=True)),
                      '', 'Artifact association, live controller use, units and physical causes require separate evidence.', ''])
    return '\n'.join(lines) + '\n'
