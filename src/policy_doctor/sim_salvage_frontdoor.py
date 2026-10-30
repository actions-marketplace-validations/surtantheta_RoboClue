"""CLI and MCP front doors for simulator-data provenance (``sim_guards``) and recording salvage (``salvage``).

Both engines stay unchanged: this module validates front-door inputs, renders their reports for people (terminal,
Markdown) and agents (bounded MCP views) and pages the complete remedy lists of large simulator reports.

* ``roboclue advanced check-sim-data SOURCE [--converted PATH]`` / MCP ``check_sim_provenance``
* ``roboclue advanced check-salvage DATASET`` / MCP ``check_recording_salvage``

Exit codes follow the rest of the CLI: 0 pass or warning, 1 error, 3 unknown, 2 when no report was produced (input
error, missing optional extra, unreadable input); ``--fail-on`` replaces the first three with 0/1.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .artifacts import InputError

SIM_COMMAND, SALVAGE_COMMAND = 'check-sim-data', 'check-salvage'
COMMANDS = (SIM_COMMAND, SALVAGE_COMMAND)
SIM_TOOL, SALVAGE_TOOL = 'check_sim_provenance', 'check_recording_salvage'
RESULT_LIMIT = 192 * 1024  # the MCP detailed bound (server.RESULT_LIMIT)
SUMMARY_LIMIT = 8 * 1024  # the MCP concise bound (server.SUMMARY_LIMIT)
TERMINAL_LIST_ENTRIES = 40  # remedy-list entries printed in full on a terminal before switching to a digest
PAGE_DEFAULT, PAGE_MAX = 500, 5000
PAGE_HEADER_BYTES = 6 * 1024  # reserved for the page envelope; entries fill the rest of RESULT_LIMIT
ENVELOPE_MARGIN = 256  # room for served_from and similar keys added after a view is bounded
SIM_MAX_ROWS, SALVAGE_MAX_ROWS, SALVAGE_MAX_FILES = 5_000_000, 1_000_000, 4096
STATUSES = ('error', 'warning', 'unknown', 'pass')
LIST_HASH_RULE = ("sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()) over the entries of "
                  "every page concatenated in offset order")


def canonical_sha256(value: Any) -> str:
    """The digest rule used for remedy lists (the same encoding as sim_guards.remedy_lists_digest)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _size(value: Any) -> int:
    return len(json.dumps(value, allow_nan=False).encode())


# ------------------------------------------------------------------------------------------------ input checks


def _local(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096 or '\x00' in value:
        raise InputError(f'{label} must be a nonempty local path')
    if '://' in value:
        raise InputError(f'{label} must be a local path; these checks never fetch over a network')
    return value


def _salvage_target(dataset: str) -> Path:
    """The dataset directory salvage reads; a non-dataset input is refused with what was expected."""
    path = Path(_local(dataset, 'dataset')).expanduser()
    if not path.exists() and not path.is_symlink():
        raise InputError(f'Dataset path does not exist: {dataset}')
    if path.name == 'info.json' and path.parent.name == 'meta' and not path.is_dir():
        path = path.parent.parent  # the alias names its dataset; a linked info.json is then refused by name
    resolved = path.resolve()
    if resolved.is_file():
        if resolved.name != 'info.json' or resolved.parent.name != 'meta':
            raise InputError(f'{dataset} is a file, not a LeRobot dataset. check-salvage reads a local LeRobot v3 '
                             'dataset directory (or its meta/info.json); robomimic or Isaac Lab HDF5 files are checked '
                             'with check-sim-data.')
        resolved = resolved.parent.parent
    elif not resolved.is_dir():
        raise InputError(f'{dataset} is not a regular directory (a FIFO, socket or device is never read)')
    info = resolved / 'meta' / 'info.json'
    if not info.exists() and not info.is_symlink():
        raise InputError(f'Not a LeRobot dataset: {resolved} has no meta/info.json. check-salvage reads native '
                         'LeRobot v3 recordings (lerobot-record writes meta/info.json when the recording starts, so a '
                         'recording without it cannot be salvaged from metadata; restore it from a backup if it was '
                         'lost). For robomimic or Isaac Lab HDF5 use check-sim-data.')
    return resolved


def _sim_input(value: Any, label: str) -> str:
    """A simulator input; a plain directory that is not a LeRobot dataset is refused with what was expected."""
    from .fs_guard import refuse_special_file

    resolved = Path(_local(value, label)).expanduser().resolve()
    if not resolved.exists():
        raise InputError(f'{label} path does not exist: {value!r}. Check the path and rerun.')
    refuse_special_file(resolved, label)
    if resolved.is_dir() and not (resolved / 'meta' / 'info.json').exists():
        raise InputError(f'{label} {value} is a directory without meta/info.json, so it is not a LeRobot dataset. '
                         'check-sim-data reads a robomimic or Isaac Lab HDF5 file, or a LeRobot dataset directory (or '
                         'its meta/info.json).')
    return value


def _strict_int(value: Any, label: str, low: int, high: int, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if type(value) is not int or not low <= value <= high:
        raise InputError(f'{label} must be an integer between {low} and {high}' + (' or null' if optional else ''))
    return value


# ------------------------------------------------------------------------------------------------ text helpers


def _ranges_text(ranges: Any, limit: int = 12) -> str:
    if not isinstance(ranges, list) or not ranges:
        return 'none'
    parts = [f'{a}-{b}' if a != b else f'{a}' for a, b in (r for r in ranges[:limit] if isinstance(r, list) and len(r) == 2)]
    return ', '.join(parts) + (f' (+{len(ranges) - limit} more ranges)' if len(ranges) > limit else '')


def _range_count(ranges: Any) -> int:
    if not isinstance(ranges, list):
        return 0
    return sum(b - a + 1 for a, b in (r for r in ranges if isinstance(r, list) and len(r) == 2))


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, allow_nan=False, default=str)
    return text if len(text) <= limit else text[:limit] + '...'


def _cut(text: str, limit: int) -> str:
    """Cut free text at a word boundary and say so."""
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(' ', 1)[0] if ' ' in text[:limit] else text[:limit]
    return head + ' [...]'


def _ordered(findings: list) -> list:
    rank = {s: i for i, s in enumerate(STATUSES)}
    return sorted((f for f in findings if isinstance(f, dict)), key=lambda f: rank.get(f.get('status'), 4))


def _finding_lines(finding: dict, *, evidence: int = 3) -> list[str]:
    lines = [f"{str(finding.get('status', 'unknown')).upper()} {finding.get('code')}", '  ' + str(finding.get('message', ''))]
    remedy = finding.get('remedy')
    if remedy and remedy != 'None.':
        lines.append('  Remedy: ' + str(remedy))
    items = finding.get('evidence') if isinstance(finding.get('evidence'), list) else []
    for item in items[:evidence]:
        if isinstance(item, dict):
            sha = str(item.get('sha256') or '')[:16]
            lines.append(f"  evidence: {item.get('path')} {item.get('pointer', '')} sha256 {sha}... value "
                         + _short(item.get('value'), 120))
    if len(items) > evidence:
        lines.append(f'  (+{len(items) - evidence} more evidence items in --json)')
    return lines


# ------------------------------------------------------------------------------------------------ simulator view


def _sim_inputs(report: dict) -> dict:
    coverage = report.get('coverage') if isinstance(report.get('coverage'), dict) else {}
    out = {}
    for role in ('source', 'converted'):
        cov = coverage.get(role)
        if not isinstance(cov, dict):
            continue
        if cov.get('kind') == 'lerobot':
            out[role] = {'path': cov.get('path'), 'kind': 'lerobot', 'info_sha256': cov.get('info_sha256'),
                         'parquet_files': cov.get('parquet_files')}
        else:
            out[role] = {'path': cov.get('path'), 'kind': 'hdf5', 'layout': cov.get('layout'),
                         'sha256': cov.get('sha256'), 'demos_total': cov.get('demos_total'),
                         'demos_inspected': cov.get('demos_inspected')}
    return out


def list_keys(report: dict) -> list[str]:
    lists = report.get('remedy_lists') if isinstance(report.get('remedy_lists'), dict) else {}
    return sorted(k for k in lists if k != 'names')


def list_digest(report: dict, max_names: int = 5, max_groups: int = 20) -> dict:
    """sim_guards.remedy_lists_digest plus, per list, the expanded entry count and the sha256 that a client can
    recompute over the pages it assembled (``LIST_HASH_RULE``)."""
    from .sim_guards import remedy_list, remedy_lists_digest

    digest = remedy_lists_digest(report, max_names=max_names, max_groups=max_groups)
    for key, part in digest['lists'].items():
        entries = remedy_list(report, key)
        part['entries'] = len(entries)
        part['list_sha256'] = canonical_sha256(entries)
        part['groups_shown'] = len(part.get('shown', []))
    return digest


def _bounded_digest(report: dict, budget: int) -> dict:
    """The largest digest that fits ``budget`` bytes; never drops a list key, its counts or its hashes."""
    for names, groups in ((5, 20), (3, 10), (2, 5), (1, 3), (0, 1)):
        digest = list_digest(report, names, groups)
        if _size(digest) <= budget:
            digest['reduced'] = (names, groups) != (5, 20)
            return digest
    digest = list_digest(report, 0, 0)
    for part in digest['lists'].values():
        part.pop('shown', None)
    digest['reduced'] = True
    return digest


def _page_hint(request: dict | None) -> str:
    args = {k: v for k, v in (request or {}).items() if v is not None}
    return (f"Complete lists: call {SIM_TOOL} again with the same inputs and remedy_list=<key> (offset 0, then each "
            f"next_offset until complete is true); {LIST_HASH_RULE} equals the list's list_sha256. Same inputs: "
            + json.dumps(args, sort_keys=True)[:600] + '. Or run `roboclue advanced check-sim-data SOURCE [--converted '
            'PATH] --json --output FILE` for the whole report.')


def human_sim(report: dict, *, complete: bool) -> str:
    lines = [f"RoboClue simulator data provenance: {str(report.get('status', 'unknown')).upper()}"]
    counts = report.get('counts') or {}
    lines.append('Findings: ' + ', '.join(f'{counts.get(s, 0)} {s}' for s in STATUSES))
    for role, item in _sim_inputs(report).items():
        if item['kind'] == 'lerobot':
            lines.append(f"{role}: {item['path']} (LeRobot dataset, {item['parquet_files']} parquet file(s)); "
                         f"meta/info.json sha256 {item['info_sha256']}")
        else:
            lines.append(f"{role}: {item['path']} ({item['layout']} HDF5, {item['demos_inspected']} of "
                         f"{item['demos_total']} demo(s) inspected); sha256 {item['sha256']}")
    coverage = report.get('coverage') if isinstance(report.get('coverage'), dict) else {}
    if coverage.get('value_budget_exhausted'):
        lines.append('The read budget was exhausted: some datasets were not assessed.')
    for finding in _ordered(report.get('findings', [])):
        lines.extend(_finding_lines(finding))
    lines.extend(_remedy_list_lines(report, complete=complete))
    lines.append('Scope: ' + str(report.get('scope', '')))
    lines.append('Use --json for every evidence value, coverage record and hash; --json, --markdown and --output FILE '
                 'contain every remedy list in full.')
    return '\n'.join(lines) + '\n'


def _remedy_list_lines(report: dict, *, complete: bool) -> list[str]:
    from .sim_guards import remedy_list

    keys = list_keys(report)
    if not keys:
        return []
    lists, names = report['remedy_lists'], report['remedy_lists'].get('names', {})
    total = sum(len(remedy_list(report, k)) for k in keys)
    full = complete or total <= TERMINAL_LIST_ENTRIES
    digest = list_digest(report)
    lines = [f"Remedy lists ({len(keys)}; the remedies above refer to them by key; sha256 {digest['sha256']} over the "
             f"canonical lists):"]
    for key in keys:
        part = digest['lists'][key]
        lines.append(f"  {key}: {part['groups']} group(s), {part['entries']} dataset entries, list sha256 "
                     f"{part['list_sha256']}")
        groups = lists[key] if full else lists[key][:5]
        for number, group in enumerate(groups, 1):
            fields = ', '.join(f'{k}={_short(v, 60)}' for k, v in group.items() if k not in ('datasets', 'demos'))
            datasets = names.get(group.get('datasets'), [])
            demos = group.get('demos')
            demo_text = ('all demos' if demos in (None, 'all') else
                         f"{len(names.get(demos, []))} demo(s): " + ', '.join(names.get(demos, []) if full else
                                                                            names.get(demos, [])[:5])
                         + ('' if full or len(names.get(demos, [])) <= 5 else ', ...'))
            shown = datasets if full else datasets[:5]
            lines.append(f"    group {number}: {fields}; {demo_text}; {len(datasets)} dataset(s): " + ', '.join(shown)
                         + ('' if len(shown) == len(datasets) else ', ...'))
        if not full and len(lists[key]) > 5:
            lines.append(f'    (+{len(lists[key]) - 5} more groups)')
    if not full:
        lines.append(f'  This terminal view abbreviates {total} entries; --json, --markdown or --output FILE writes '
                     'every list in full (policy_doctor.sim_guards.remedy_list(report, key) expands one).')
    return lines


def markdown_sim(report: dict) -> str:
    from .reports import _text
    from .sim_guards import remedy_list

    lines = ['# RoboClue simulator data provenance', '',
             f"**Status:** {_text(str(report.get('status', 'unknown')).upper())}", '',
             '| status | count |', '|---|---|']
    counts = report.get('counts') or {}
    lines += [f'| {s} | {counts.get(s, 0)} |' for s in STATUSES] + ['', '## Inputs', '']
    for role, item in _sim_inputs(report).items():
        lines.append(f'- **{role}**: {_text(item["path"])}; ' + '; '.join(
            f'{_text(k)}: {_text(v)}' for k, v in item.items() if k != 'path'))
    lines += ['', '## Findings', '']
    for finding in _ordered(report.get('findings', [])):
        lines += [f"### {_text(str(finding.get('status', '')).upper())} {_text(finding.get('code'))}", '',
                  _text(finding.get('message', '')), '']
        if finding.get('remedy') and finding.get('remedy') != 'None.':
            lines += ['**Remedy:** ' + _text(finding['remedy']), '']
        for item in finding.get('evidence', []) if isinstance(finding.get('evidence'), list) else []:
            if isinstance(item, dict):
                lines.append(f"- evidence {_text(item.get('path'))} pointer {_text(item.get('pointer', ''))} sha256 "
                             f"{_text(item.get('sha256'))} value {_text(_short(item.get('value'), 400))}")
        lines.append('')
    keys = list_keys(report)
    if keys:
        digest = list_digest(report)
        lines += ['## Remedy lists (complete)', '',
                  f"sha256 of the canonical lists: {_text(digest['sha256'])}. One line per dataset entry, as "
                  'policy\\_doctor.sim\\_guards.remedy\\_list(report, key) expands it.', '']
        for key in keys:
            part = digest['lists'][key]
            lines += [f'### {_text(key)}', '', f"{part['entries']} entries in {part['groups']} group(s); list sha256 "
                      f"{_text(part['list_sha256'])}", '']
            for entry in remedy_list(report, key):
                lines.append('- ' + _text(json.dumps(entry, sort_keys=True)))
            lines.append('')
    lines += ['## Scope', '', _text(report.get('scope', '')), '']
    return '\n'.join(lines)


def _shown(findings: list, message: int | None, remedy: int | None, evidence: int) -> list[dict]:
    out = []
    for finding in findings:
        item = {'code': finding.get('code'), 'status': finding.get('status')}
        cut = []
        for key, limit in (('message', message), ('remedy', None)):
            text = str(finding.get(key, ''))
            if key == 'remedy' and remedy == -1:
                item[key] = 'Full remedy omitted from this bounded view; use the CLI with --json --output FILE. [...]'
                item['remedy_omitted'] = True
            else:
                item[key] = text if limit is None else _cut(text, limit)
            if item[key] != text:
                cut.append(key)
                item[f'full_{key}_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        if cut:
            item['text_truncated'] = cut
        item['evidence'] = [_witness(e) for e in (finding.get('evidence') or [])[:evidence] if isinstance(e, dict)]
        out.append(item)
    return out


def _witness(evidence: dict) -> dict:
    """One evidence item, bounded: the value as short JSON text, long paths and pointers cut with a marker."""
    out = {}
    for key in ('path', 'pointer', 'sha256', 'value'):
        if key in evidence:
            value = evidence[key]
            out[key] = _short(value, 200) if key == 'value' else _short(value, 512) if isinstance(value, str) else value
    return out


def _concise(report: dict, extra: dict, next_text: str, *, extra_shrink=None, priority_codes=()) -> dict:
    """A verdict-preserving view within SUMMARY_LIMIT. Shown findings keep their full remedy as long as possible;
    every cut text is marked with the sha256 of its full text; no verdict, count or list key is ever dropped."""
    findings = report.get('findings') if isinstance(report.get('findings'), list) else []
    attention = [f for f in _ordered(findings) if f.get('status') != 'pass']
    # A withheld recovery plan must retain its reason even when repeated
    # per-feature errors fill the preview. Overall severity/counts stay intact.
    if priority_codes:
        priorities = {code: position for position, code in enumerate(priority_codes)}
        attention.sort(key=lambda f: priorities.get(f.get('code'), len(priorities)))
    passed = [str(f.get('code'))[:80] for f in findings if isinstance(f, dict) and f.get('status') == 'pass']
    base = {'summary': True, 'kind': report.get('kind'), 'status': report.get('status'),
            'counts': report.get('counts'), 'findings_total': len(findings),
            'passed_checks': passed[:16], 'passed_checks_omitted': max(0, len(passed) - 16)}
    base.update(extra)
    base['scope'] = str(report.get('scope', ''))
    base['full_report_sha256'] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    stages = [(None, None, 2, 12), (600, None, 2, 12), (400, 900, 2, 10), (300, 600, 1, 8), (240, 400, 1, 6),
              (200, 300, 1, 4), (160, 240, 1, 3), (120, 200, 0, 3), (120, 200, 0, 1),
              (120, -1, 0, 1), (120, -1, 0, 0)]
    for shrink in (False, True):
        if shrink and extra_shrink is not None:
            base.update(extra_shrink())
        for message, remedy, evidence, count in stages:
            view = dict(base, findings_shown=_shown(attention[:count], message, remedy, evidence),
                        findings_omitted=max(0, len(attention) - count), next=next_text)
            # FastMCP renders dict results as indented text as well as structured
            # content. Budget that representation, not just compact JSON.
            if len(json.dumps(view, indent=2, allow_nan=False).encode()) <= SUMMARY_LIMIT - ENVELOPE_MARGIN:
                return view
    raise InputError('The summary header exceeds 8 KiB; use the matching CLI command with --json --output FILE')


def concise_sim(report: dict, request: dict | None = None) -> dict:
    keys = list_keys(report)
    extra = {'inputs': _sim_inputs(report)}
    if keys:
        extra['remedy_lists'] = _bounded_digest(report, 3 * 1024)
        extra['remedy_lists_complete_in'] = _page_hint(request)
    return _concise(report, extra,
                    "Request response_format='detailed' for full finding text, evidence and coverage"
                    + (', and remedy_list pages for the complete remedy lists' if keys else '')
                    + ', or run roboclue advanced check-sim-data ... --json --output FILE.',
                    extra_shrink=(lambda: {'remedy_lists': _bounded_digest(report, 1024)}) if keys else None)


def detailed_sim(report: dict, request: dict | None = None) -> dict:
    """The full report when it fits RESULT_LIMIT. Otherwise every finding in full with coverage and the remedy-list
    digest, and remedy_lists replaced by an explicit pointer to the pages (never silently dropped)."""
    size = _size(report)
    if size <= RESULT_LIMIT - ENVELOPE_MARGIN:
        return report
    view = {k: report.get(k) for k in ('schema_version', 'kind', 'status', 'counts', 'findings', 'scope')}
    view.update(detailed_bounded=True, truncated_from_bytes=size, result_limit_bytes=RESULT_LIMIT,
                remedy_lists=None, remedy_lists_omitted=list_keys(report),
                remedy_lists_digest=_bounded_digest(report, 32 * 1024), remedy_lists_complete_in=_page_hint(request),
                full_report_sha256=hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'),
                    ensure_ascii=False, allow_nan=False).encode()).hexdigest())
    view['coverage'] = report.get('coverage')
    if _size(view) > RESULT_LIMIT - ENVELOPE_MARGIN:
        view['coverage'] = {'omitted': True, 'inputs': _sim_inputs(report)}
        view['coverage_omitted'] = True
    if _size(view) > RESULT_LIMIT - ENVELOPE_MARGIN:
        concise = concise_sim(report, request)
        concise.update(detailed_bounded=True, truncated_from_bytes=size, remedy_lists_omitted=list_keys(report),
                       next='The detailed report exceeds the 192 KiB response bound even without its remedy lists; '
                            'use remedy_list pages for the lists and roboclue advanced check-sim-data ... --json --output '
                            'FILE for every finding.')
        return concise
    return view


def remedy_page(report: dict, key: Any, offset: Any, limit: Any) -> dict:
    """One bounded page of remedy_list(report, key): entries [offset, offset + returned)."""
    from .sim_guards import remedy_list

    keys = list_keys(report)
    if not isinstance(key, str) or key not in keys:
        raise InputError(f'remedy_list must name a list of this report; available: {", ".join(keys) or "none"}')
    offset = _strict_int(0 if offset is None else offset, 'offset', 0, 10 ** 8)
    limit = _strict_int(PAGE_DEFAULT if limit is None else limit, 'limit', 1, PAGE_MAX)
    entries = remedy_list(report, key)
    if offset > len(entries):
        raise InputError(f'offset {offset} is past the end of {key} ({len(entries)} entries)')
    budget, used, page = RESULT_LIMIT - PAGE_HEADER_BYTES, 0, []
    for entry in entries[offset:offset + limit]:
        cost = _size(entry) + 2
        if used + cost > budget:
            break
        page.append(entry)
        used += cost
    if not page and offset < len(entries):
        raise InputError(f'Entry {offset} of {key} alone exceeds the {RESULT_LIMIT}-byte response bound; run '
                         'roboclue advanced check-sim-data ... --json --output FILE for the complete report')
    end = offset + len(page)
    digest = list_digest(report, 0, 0)
    return {'kind': 'sim_provenance_remedy_list_page', 'status': report.get('status'), 'remedy_list': key,
            'offset': offset, 'limit': limit, 'returned': len(page), 'entries_total': len(entries),
            'next_offset': end if end < len(entries) else None, 'complete': end >= len(entries),
            'byte_bounded': len(page) < min(limit, len(entries) - offset),
            'list_sha256': canonical_sha256(entries), 'remedy_lists_sha256': digest['sha256'],
            'inputs': {role: item.get('sha256') or item.get('info_sha256') for role, item in _sim_inputs(report).items()},
            'verify': f'{LIST_HASH_RULE} equals list_sha256. Pages whose remedy_lists_sha256 differ come from different '
                      'reads; restart at offset 0.',
            'entries': page}


# ------------------------------------------------------------------------------------------------ salvage view


def _plan_summary(plan: Any) -> dict:
    if not isinstance(plan, dict):
        return {'strategy': None}
    files = plan.get('files') if isinstance(plan.get('files'), dict) else {}
    validity = plan.get('validity') if isinstance(plan.get('validity'), dict) else {}
    out = {'strategy': plan.get('strategy'), 'applied': False,
           'note': 'A description for a separate copy; RoboClue never applies it and wrote nothing.'}
    for key in ('reason', 'kept_episodes', 'kept_frames', 'corrected_info', 'changes'):
        if key in plan:
            out[key] = _cut(str(plan[key]), 400) if key == 'reason' else plan[key]
    if files:
        out['files'] = {k: files.get(k) for k in ('copy_unchanged_count', 'rewrite_count', 'omit_count',
                                                   'lists_complete')}
    if validity:
        out['resume_safe'] = validity.get('resume_safe')
        out['resume_blockers'] = [_cut(str(b), 240) for b in (validity.get('resume_blockers') or [])[:3]]
        out['resume_safety_scope'] = validity.get('resume_safe_applies_to',
            'Prediction for a separate copy after the entire plan; not current-directory permission.')
        out['copy_verification_required'] = True
        out['current_directory_resume_status'] = validity.get('current_directory_resume_status', 'unknown')
    if plan.get('required_data_rewrite'):
        out['required_data_rewrite'] = plan['required_data_rewrite'][:2]
    return out


def _episodes_summary(report: dict, drops: int = 8) -> dict:
    episodes = report.get('episodes') if isinstance(report.get('episodes'), dict) else {}
    drop = episodes.get('drop') if isinstance(episodes.get('drop'), list) else []
    return {'keep': (episodes.get('keep') or [])[:16], 'keep_count': _range_count(episodes.get('keep')),
            'keep_scope': episodes.get('keep_scope', 'Episodes selected by the separate-copy plan; not original-directory approval.'),
            'exclusion_scope': 'drop lists mean excluded from this metadata-only copy; never delete original bytes.',
            'recovery_candidates': (episodes.get('recovery_candidates') or [])[:8],
            'recovery_candidates_omitted': max(0, len(episodes.get('recovery_candidates') or []) - 8),
            'drop_count': (report.get('counts') or {}).get('drop', len(drop)),
            'drop': [{'episode_index': d.get('episode_index'),
                      'reasons': [str(r)[:60] for r in (d.get('reasons') or [])[:4]] if isinstance(d.get('reasons'), list)
                      else _short(d.get('reasons'), 120)} for d in drop[:drops] if isinstance(d, dict)],
            'drop_omitted': max(0, len(drop) - drops) + (1 if episodes.get('drop_truncated') else 0),
            'need_renumbering': (episodes.get('need_renumbering') or [])[:8],
            'unresolved': (episodes.get('unresolved') or [])[:16],
            'unresolved_count': _range_count(episodes.get('unresolved')),
            'committed_but_not_verifiable': (episodes.get('committed_but_not_verifiable') or [])[:8],
            'staging_only': (episodes.get('staging_only') or [])[:8]}


def _reason_text(reasons: Any) -> str:
    if isinstance(reasons, list):
        return ', '.join(str(r) if not isinstance(r, dict) else _short(r, 80) for r in reasons)
    return _short(reasons, 120)


def human_salvage(report: dict) -> str:
    coverage = report.get('coverage') if isinstance(report.get('coverage'), dict) else {}
    counts = (report.get('counts') or {}).get('findings') or {}
    episodes = report.get('episodes') if isinstance(report.get('episodes'), dict) else {}
    lines = [f"RoboClue recording salvage: {str(report.get('status', 'unknown')).upper()}",
             f"Reads bytes only; nothing was written. Reference: LeRobot "
             f"{str((report.get('upstream_reference') or {}).get('revision', ''))[:7]} (v3 datasets).",
             'Findings: ' + ', '.join(f'{counts.get(s, 0)} {s}' for s in STATUSES),
             f"Coverage complete: {'yes' if coverage.get('complete') else 'no'}"]
    if episodes:
        drop = episodes.get('drop') if isinstance(episodes.get('drop'), list) else []
        lines.append(f"Episodes to keep: {_ranges_text(episodes.get('keep'))} ({_range_count(episodes.get('keep'))})")
        if drop:
            lines.append(f"Episodes excluded from this metadata-only copy: {len(drop)}" + (' (first listed; more in --json)'
                                                              if episodes.get('drop_truncated') else ''))
            for item in drop[:16]:
                if isinstance(item, dict):
                    lines.append(f"  episode {item.get('episode_index')}: {_reason_text(item.get('reasons'))}")
            if len(drop) > 16:
                lines.append(f'  (+{len(drop) - 16} more in --json)')
            lines.append('Keep the original bytes: exclusion from this plan is not a deletion instruction.')
        if episodes.get('recovery_candidates'):
            lines.append('Recovery candidates outside this plan: ' + _ranges_text(episodes['recovery_candidates'])
                         + ' (rows and frame evidence survive; decoding/regenerated metadata not verified)')
        for key, label in (('need_renumbering', 'Episodes needing renumbering'),
                           ('unresolved', 'Episodes awaiting review; no exclusion approved'),
                           ('committed_but_not_verifiable', 'Committed but not verifiable')):
            if episodes.get(key):
                lines.append(f'{label}: {_ranges_text(episodes[key])}')
        if episodes.get('staging_only'):
            staged = episodes['staging_only']
            lines.append('Staging frames only (never saved): ' + ', '.join(map(str, staged[:16]))
                         + (f' ... (+{len(staged) - 16} more in --json)' if len(staged) > 16 else ''))
    plan = _plan_summary(report.get('plan'))
    if plan.get('strategy'):
        text = f"Salvage plan (description only, never applied): {plan['strategy']}"
        if plan.get('reason'):
            text += f" - {plan['reason']}"
        if 'kept_episodes' in plan:
            text += f"; keeps {plan['kept_episodes']} episode(s), {plan.get('kept_frames')} frame(s)"
        files = plan.get('files') or {}
        if files:
            text += (f"; copy {files.get('copy_unchanged_count')} file(s) unchanged, rewrite {files.get('rewrite_count')}, "
                     f"omit {files.get('omit_count')}")
        if 'resume_safe' in plan:
            text += ('; planned-copy resume: no listed blocker (prediction only, not applied or verified)'
                     if plan['resume_safe'] else '; planned-copy resume safe: no')
        lines.append(text)
        current_resume = plan.get('current_directory_resume_status', 'unknown')
        lines.append('Current-directory resume: ' + ('UNSAFE; do not resume here.' if current_resume == 'unsafe'
                     else 'no hazard detected in covered checks; verify with the native loader.'
                     if current_resume == 'no_hazard_detected' else 'not established.'))
        if plan.get('files', {}).get('lists_complete') is False:
            lines.append('Plan file lists are truncated; do not apply this partial listing.')
        for rewrite in plan.get('required_data_rewrite', []):
            lines.append(f"  Separate-copy data rewrite needed: {rewrite['file']}; retain physical rows "
                         f"{rewrite['keep_physical_rows']} (end exclusive), excluding {rewrite['rows_to_exclude']} "
                         'unreferenced row(s). Preserve schema metadata, then recheck the copy and native loader.')
        for blocker in plan.get('resume_blockers', []):
            lines.append('  resume blocker: ' + blocker)
    for finding in _ordered(report.get('findings', [])):
        lines.extend(_finding_lines(finding, evidence=2))
    lines.append('Scope: ' + str(report.get('scope', '')))
    lines.append('Use --json for the complete plan (file lists, info.json fields, pointer corrections) and per-episode '
                 'evidence, or --markdown for a portable report.')
    return '\n'.join(lines) + '\n'


def markdown_salvage(report: dict) -> str:
    from .reports import _text

    lines = ['# RoboClue recording salvage', '', f"**Status:** {_text(str(report.get('status', 'unknown')).upper())}",
             '', 'Reads bytes only; nothing was written. The plan is a description for a separate copy.', '']
    lines += ['## Verdict', ''] + [_text(line) for line in human_salvage(report).splitlines()[1:] if line] + ['']
    plan = report.get('plan')
    if isinstance(plan, dict):
        complete = (plan.get('files') or {}).get('lists_complete') is True
        title = '## Plan (complete)' if complete else '## Plan (as reported; inspect files.lists_complete)'
        lines += [title, '', '```json', json.dumps(plan, indent=2, sort_keys=True, allow_nan=False)
                  .replace('```', '` ` `'), '```', '']
    episodes = report.get('episodes')
    if isinstance(episodes, dict):
        lines += ['## Episodes (complete ledger as reported)', '', '```json',
                  json.dumps(episodes, indent=2, sort_keys=True, allow_nan=False).replace('```', '` ` `'), '```', '']
    lines += ['## Artifacts', '']
    for item in report.get('artifacts', []) if isinstance(report.get('artifacts'), list) else []:
        if isinstance(item, dict):
            lines.append(f"- {_text(item.get('path'))} sha256 {_text(item.get('sha256'))}")
    return '\n'.join(lines) + '\n'


def concise_salvage(report: dict) -> dict:
    coverage = report.get('coverage') if isinstance(report.get('coverage'), dict) else {}
    extra = {'counts_episodes': {k: v for k, v in (report.get('counts') or {}).items() if k != 'findings'},
             'episodes': _episodes_summary(report), 'plan': _plan_summary(report.get('plan')),
             'coverage': {'complete': coverage.get('complete'), 'video_index_status': coverage.get('video_index_status'),
                          'max_rows': coverage.get('max_rows'), 'max_files': coverage.get('max_files')}}
    extra['counts'] = (report.get('counts') or {}).get('findings')
    extra['independent_error_codes'] = list(dict.fromkeys(
        f['code'] for f in report.get('findings', []) if f.get('status') == 'error'))[:16]

    def smaller():
        return {'episodes': _episodes_summary(report, drops=3), 'plan': {k: v for k, v in _plan_summary(
            report.get('plan')).items() if k not in ('changes', 'corrected_info')}}

    return _concise(report, extra, "Request response_format='detailed' for the complete plan, per-episode ledger and "
                    'evidence, or run roboclue advanced check-salvage DATASET --json --output FILE.', extra_shrink=smaller,
                    priority_codes=('salvage.metadata_row_conflict', 'salvage.resume_hazard', 'salvage.tasks_unusable',
                                    'salvage.row_identity', 'salvage.video_conflict', 'salvage.video_file_truncated'))


def detailed_salvage(report: dict) -> dict:
    size = _size(report)
    if size <= RESULT_LIMIT:
        return report
    view = concise_salvage(report)
    view.update(detailed_bounded=True, truncated_from_bytes=size,
                next='The full salvage report exceeds the 192 KiB response bound; run roboclue advanced check-salvage '
                     'DATASET --json --output FILE for the complete report.')
    return view


# ------------------------------------------------------------------------------------------------ CLI


def add_cli_commands(commands: Any) -> None:
    sim_help = ('Check simulator demonstration provenance offline: Isaac Lab quaternion order/format_version, robomimic '
                'controller vs actions, LeRobot quaternion names')
    sim = commands.add_parser(SIM_COMMAND, help=sim_help, description=sim_help + '. Reads HDF5 attributes and selected '
                              'numeric datasets only; no simulator, converter or model runs. See docs/SIM_DATA_PROVENANCE.md.')
    sim.add_argument('source', help='robomimic or Isaac Lab HDF5 file, or a LeRobot dataset directory (or its '
                     'meta/info.json) when no HDF5 exists')
    sim.add_argument('--converted', help='Optional HDF5 derived from SOURCE (merge, convert or augment output) or the '
                     'LeRobot dataset converted from it')
    sim.add_argument('--max-demos', type=int, help='Inspect at most N demos per file, evenly spaced, 1-4096 (default: '
                     'every demo); partial coverage never passes')
    sim.add_argument('--max-rows', type=int, default=200000, help=f'LeRobot rows read, 1-{SIM_MAX_ROWS}')
    salvage_help = ('Decide which episodes of an interrupted local LeRobot v3 recording to keep or drop; describes a '
                    'metadata-only plan and writes nothing')
    salvage = commands.add_parser(SALVAGE_COMMAND, help=salvage_help, description=salvage_help + '. The plan is a '
                                  'description for a separate copy and is never applied. See docs/RECORDING_SALVAGE.md.')
    salvage.add_argument('dataset', help='Local LeRobot v3 dataset directory (or its meta/info.json)')
    salvage.add_argument('--video-headers', action=argparse.BooleanOptionalAction, default=True,
                         help='Read MP4 header boxes of local videos and encoder leftovers (default); opting out leaves '
                              'video coverage unknown')
    salvage.add_argument('--max-rows', type=int, default=200000, help=f'Data rows read, 1-{SALVAGE_MAX_ROWS}')
    salvage.add_argument('--max-files', type=int, default=1024,
                         help=f'Native Parquet intake and separate video-file inventory bounds, 1-{SALVAGE_MAX_FILES}')
    outputs = ((sim, 'Write this report to a new file instead of stdout; a file always holds every remedy list in full'),
               (salvage, 'Write this report to a new file instead of stdout, never inside the checked dataset; --json '
                         'or --markdown hold the complete plan'))
    for command, output_help in outputs:
        formats = command.add_mutually_exclusive_group()
        formats.add_argument('--json', action='store_true', help='Print the complete machine-readable report')
        formats.add_argument('--markdown', action='store_true', help='Create a portable evidence report')
        command.add_argument('--output', type=Path, help=output_help)
        command.add_argument('--fail-on', choices=('error', 'warning', 'unknown'),
                             help='CI gate: exit 1 if any finding has this status (warning also counts errors; '
                                  'unknown counts every non-pass), otherwise 0. Default keeps the 0/1/3 codes')


def _refuse_output_inside(output: Path, protected: list[Path]) -> None:
    target = output.expanduser().resolve()
    for root in protected:
        if root.is_file() and root.name == 'info.json' and root.parent.name == 'meta':
            root = root.parent.parent  # a LeRobot info.json alias names its dataset directory
        if root.is_dir() and target.is_relative_to(root):
            raise InputError(f'--output {output} lies inside the checked input {root}; write the report elsewhere so '
                             'the checked bytes stay unchanged')


def _render(result: dict, args: Any) -> str:
    sim = args.command == SIM_COMMAND
    if args.json:
        return json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.markdown:
        return markdown_sim(result) if sim else markdown_salvage(result)
    return human_sim(result, complete=args.output is not None) if sim else human_salvage(result)


def run_cli(args: Any) -> int:
    """Run one front-door command with the CLI's output modes and exit codes."""
    try:
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise InputError(f'--output {args.output} already exists; reports are written only to new files')
        if args.command == SIM_COMMAND:
            from .sim_guards import check_sim_provenance

            source = _sim_input(args.source, 'SOURCE')
            converted = _sim_input(args.converted, '--converted') if args.converted is not None else None
            protected = [Path(p).expanduser().resolve() for p in (source, converted) if p is not None]
            if args.output is not None:
                _refuse_output_inside(args.output, protected)
            result = check_sim_provenance(source, converted=converted, max_demos=args.max_demos,
                                          max_rows=args.max_rows)
        else:
            from .salvage import check_salvage

            root = _salvage_target(args.dataset)
            if args.output is not None:
                _refuse_output_inside(args.output, [root])
            result = check_salvage(str(root), max_rows=args.max_rows, max_files=args.max_files,
                                   video_headers=args.video_headers)
        rendered = _render(result, args)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(rendered)
        else:
            sys.stdout.write(rendered)
        if args.fail_on:
            failing = {'error': {'error'}, 'warning': {'error', 'warning'},
                       'unknown': {'error', 'warning', 'unknown'}}[args.fail_on]
            return 1 if any(f.get('status') in failing for f in result.get('findings', [])) else 0
        return {'error': 1, 'unknown': 3}.get(result.get('status'), 0)
    except (InputError, OSError, ImportError) as exc:
        sys.stderr.write(json.dumps({'error': {'code': 'invalid_input', 'message': str(exc)}}) + '\n')
        return 2
    except (ValueError, TypeError, KeyError, RuntimeError, RecursionError) as exc:  # never a bare traceback
        sys.stderr.write(json.dumps({'error': {'code': 'check_failed', 'message':
            f'{type(exc).__name__}: {exc}'[:2000] + '. No report was produced; please report this input.'}}) + '\n')
        return 2


# ------------------------------------------------------------------------------------------------ MCP


class ReportCache:
    """The last few complete simulator reports, so pages and other views need not repeat a long check. An entry is used
    only while every input file has the same device, inode, size, modification and change time as when it was read;
    directory inputs are never cached."""

    def __init__(self, size: int = 4):
        self.size, self.lock, self.items = size, threading.Lock(), OrderedDict()

    @staticmethod
    def identity(paths: list[str]) -> tuple | None:
        out = []
        for path in paths:
            # A LeRobot info.json alias depends on other metadata and data shards. Its stat identity alone
            # cannot validate their contents. Cache only complete HDF5-file inputs, never dataset aliases.
            if Path(path).suffix.lower() not in ('.h5', '.hdf5'):
                return None
            try:
                info = os.stat(path, follow_symlinks=False)
            except OSError:
                return None
            if not os.path.isfile(path) or os.path.islink(path):
                return None
            out.append((path, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        return tuple(out)

    def get(self, key, identity):
        with self.lock:
            item = self.items.get(key)
            if item is None or identity is None or item[0] != identity:
                return None
            self.items.move_to_end(key)
            return item[1]

    def put(self, key, identity, report):
        if identity is None:
            return
        with self.lock:
            self.items[key] = (identity, report)
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)


def register_mcp_tools(server: Any, *, guard: Any, allowed: list[Path], annotations: Any) -> None:
    """Add check_sim_provenance and check_recording_salvage to a FastMCP server (both read-only, default set)."""
    from functools import partial

    from .server import Flag, ResponseFormat, _nullable, _typed

    roots = [str(root) for root in allowed]
    cache = ReportCache()
    MaxDemos = _typed(_nullable({'type': 'integer', 'minimum': 1, 'maximum': 4096}),
                      'Demos inspected per file, evenly spaced, 1-4096; null (default) inspects every demo. Partial '
                      'coverage never passes.')
    SimRows = _typed({'type': 'integer', 'minimum': 1, 'maximum': SIM_MAX_ROWS}, f'LeRobot rows read, 1-{SIM_MAX_ROWS}.')
    ListKey = _typed(_nullable({'type': 'string', 'minLength': 1, 'maxLength': 200}),
                     "A key of the report's remedy_lists (listed in the concise digest); returns one bounded page of "
                     'its complete expanded entries instead of the report.')
    Offset = _typed(_nullable({'type': 'integer', 'minimum': 0, 'maximum': 10 ** 8}),
                    'First entry of the page (default 0); use the previous next_offset. Requires remedy_list.')
    Limit = _typed(_nullable({'type': 'integer', 'minimum': 1, 'maximum': PAGE_MAX}),
                   f'Entries per page, 1-{PAGE_MAX} (default {PAGE_DEFAULT}); a page also stops at the 192 KiB bound. '
                   'Requires remedy_list.')
    SalvageRows = _typed({'type': 'integer', 'minimum': 1, 'maximum': SALVAGE_MAX_ROWS},
                         f'Data rows read, 1-{SALVAGE_MAX_ROWS}.')
    SalvageFiles = _typed({'type': 'integer', 'minimum': 1, 'maximum': SALVAGE_MAX_FILES},
                          f'Parquet files read, 1-{SALVAGE_MAX_FILES}.')

    def sim_call(source, converted, max_demos, max_rows, response_format, remedy_list, offset, limit):
        from .sim_guards import check_sim_provenance as run_check

        if response_format not in ('concise', 'detailed'):
            raise InputError("response_format must be 'concise' or 'detailed'")
        if remedy_list is None and (offset is not None or limit is not None):
            raise InputError('offset and limit select a remedy-list page and require remedy_list')
        _strict_int(max_demos, 'max_demos', 1, 4096, optional=True)
        _strict_int(max_rows, 'max_rows', 1, SIM_MAX_ROWS)
        if remedy_list is not None:
            if not isinstance(remedy_list, str) or not remedy_list:
                raise InputError('remedy_list must be a nonempty list key')
            _strict_int(0 if offset is None else offset, 'offset', 0, 10 ** 8)
            _strict_int(PAGE_DEFAULT if limit is None else limit, 'limit', 1, PAGE_MAX)
        checked = [_sim_input(guard(_local(source, 'source')), 'source')]
        if converted is not None:
            checked.append(_sim_input(guard(_local(converted, 'converted')), 'converted'))
        request = {'source': source, 'converted': converted, 'max_demos': max_demos,
                   'max_rows': max_rows if max_rows != 200000 else None}
        key = (tuple(checked), max_demos, max_rows)
        identity = cache.identity(checked)
        report = cache.get(key, identity)
        served = 'cached_report_same_file_identity' if report is not None else 'fresh_read'
        if report is None:
            report = run_check(checked[0], converted=checked[1] if len(checked) > 1 else None, read_roots=roots,
                               max_demos=max_demos, max_rows=max_rows)
            if identity is not None and cache.identity(checked) == identity:
                cache.put(key, identity, report)
        if remedy_list is not None:
            return dict(remedy_page(report, remedy_list, offset, limit), served_from=served)
        view = concise_sim(report, request) if response_format == 'concise' else detailed_sim(report, request)
        return dict(view, served_from=served)  # a shallow copy: the cached report is never changed

    def salvage_call(dataset, video_headers, max_rows, max_files, response_format):
        from .salvage import check_salvage

        if response_format not in ('concise', 'detailed'):
            raise InputError("response_format must be 'concise' or 'detailed'")
        if type(video_headers) is not bool:
            raise InputError('video_headers must be a strict boolean')
        _strict_int(max_rows, 'max_rows', 1, SALVAGE_MAX_ROWS)
        _strict_int(max_files, 'max_files', 1, SALVAGE_MAX_FILES)
        guard(_local(dataset, 'dataset'))  # authorize the path as given before anything is read
        root = guard(str(_salvage_target(dataset)))
        report = check_salvage(root, max_rows=max_rows, max_files=max_files, video_headers=video_headers)
        return concise_salvage(report) if response_format == 'concise' else detailed_salvage(report)

    async def in_thread(function):
        import anyio

        # A long or slow input runs off the event loop, so later requests are still answered.
        return await anyio.to_thread.run_sync(function)

    @server.tool(name=SIM_TOOL, annotations=annotations)
    async def check_sim_provenance(source: str, converted: str | None = None, max_demos: MaxDemos = None,
                                   max_rows: SimRows = 200000, response_format: ResponseFormat = 'concise',
                                   remedy_list: ListKey = None, offset: Offset = None,
                                   limit: Limit = None) -> dict[str, Any]:
        """Check simulator demonstration provenance offline: a robomimic/Isaac Lab HDF5 file (or a LeRobot dataset when no HDF5 exists), optionally against converted = an HDF5 derived from it (merge/convert/augment output) or the LeRobot dataset converted from it. Guards Isaac Lab quaternion order and format_version (v3.0.0), robomimic env_args controller vs recorded actions (robosuite 1.4 control_delta, 1.5 input_type/input_ref_frame) and LeRobot features named as quaternions. Reads HDF5 attributes and selected numeric datasets only; no simulator, converter or model runs. Every demo is inspected unless max_demos caps it; partial coverage is never a pass; undecidable quaternion orders stay unknown or conditional. Concise (default) keeps the verdict, findings that need attention and a digest of report['remedy_lists'] (per list: groups, entries, list_sha256). Detailed returns the full report up to 192 KiB; a larger report keeps every finding and replaces remedy_lists with the digest. Complete lists: pass remedy_list=<key> (offset, limit) for bounded pages of remedy_list(report, key); sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()) over all pages equals list_sha256. Repeated calls with the same inputs and bounds reuse the server's last report only while every input file is unchanged (device, inode, size, mtime, ctime); otherwise they read again; served_from says which. Both paths must lie inside the read roots."""
        return await in_thread(partial(sim_call, source, converted, max_demos, max_rows, response_format,
                                       remedy_list, offset, limit))

    @server.tool(name=SALVAGE_TOOL, annotations=annotations)
    async def check_recording_salvage(dataset: str, video_headers: Flag = True, max_rows: SalvageRows = 200000,
                                      max_files: SalvageFiles = 1024,
                                      response_format: ResponseFormat = 'concise') -> dict[str, Any]:
        """Decide which episodes of an interrupted local LeRobot v3 recording (lerobot-record killed, Ctrl-C during encoding, full disk) are complete and which must be dropped and why, and describe a bounded metadata-only salvage plan for a separate copy. Reads bytes only (info/stats JSON, Parquet footers, identity columns, staging and encoder leftovers, optional MP4 header boxes); writes nothing and never applies the plan. Pinned LeRobot 146d847. Footer-less Parquet is never parsed; no video is decoded. A symlink inside the dataset is refused with the link named (check a real copy made with cp -rL). Concise (default) keeps the verdict, keep/drop episodes, plan strategy and resume safety; detailed returns the complete report (bounded to 192 KiB). The dataset must lie inside the read roots."""
        return await in_thread(partial(salvage_call, dataset, video_headers, max_rows, max_files, response_format))
