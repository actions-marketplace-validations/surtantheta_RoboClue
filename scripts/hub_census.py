#!/usr/bin/env python3
"""Real-world census of the `check` front door on popular public LeRobot policies.

select : apply the frozen selection rule (docs/real_world/CENSUS_PROTOCOL_A_20261022.md)
run    : run the product engine on one set of selected rows
summary: aggregate one run directory
Selection and pairing use only public JSON. The product engine is imported unmodified.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path

API = 'https://huggingface.co/api'
LISTING = f'{API}/models?filter=lerobot&sort=downloads&direction=-1&limit=500'


def get(url: str, limit: int = 8 * 1024 * 1024) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': 'RoboClue-census/1', 'Accept': 'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError(f'response over {limit} bytes: {url}')
            return data
        except (OSError, ValueError) as exc:
            if attempt == 2 or isinstance(exc, ValueError):
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select(out: Path, dev: int, heldout: int) -> None:
    out.mkdir(parents=True, exist_ok=False)
    listing = get(LISTING)
    (out / 'listing.json').write_bytes(listing)
    entries = json.loads(listing)
    lerobot = sum(e.get('library_name') == 'lerobot' for e in entries)
    if not entries or lerobot < 0.8 * len(entries):
        raise SystemExit(f'Listing is not a LeRobot listing ({lerobot}/{len(entries)} entries); refusing to select')
    rows, skipped = [], []
    for index, entry in enumerate(entries):
        if len(rows) >= dev + heldout:
            break
        model = entry['id']
        if entry.get('library_name') != 'lerobot':
            skipped.append({'model': model, 'reason': f"library_name is {entry.get('library_name')!r}"})
            continue
        time.sleep(0.3)
        try:
            info_bytes = get(f'{API}/models/{model}/revision/main')
            info = json.loads(info_bytes)
        except Exception as exc:  # recorded, never silently replaced
            skipped.append({'model': model, 'reason': f'revision API failed: {type(exc).__name__}: {exc}'})
            continue
        listed = {s.get('rfilename') for s in info.get('siblings', []) if isinstance(s, dict)}
        if info.get('private') is not False or info.get('gated') not in (None, False):
            skipped.append({'model': model, 'reason': 'private or gated'})
            continue
        if 'config.json' not in listed:
            skipped.append({'model': model, 'reason': 'no config.json'})
            continue
        commit = info['sha']
        row = {'rank': len(rows), 'listing_index': index, 'model': model,
               'model_commit': commit, 'downloads': entry.get('downloads'), 'likes': entry.get('likes'),
               'revision_api_sha256': sha(info_bytes), 'dataset': None, 'pairing': None}
        if 'train_config.json' in listed:
            try:
                raw = get(f'https://huggingface.co/{model}/resolve/{commit}/train_config.json', 2 * 1024 * 1024)
                repo_id = json.loads(raw).get('dataset', {}).get('repo_id')
                row['train_config_sha256'] = sha(raw)
                if isinstance(repo_id, str) and repo_id.count('/') == 1:
                    row['dataset'], row['pairing'] = repo_id, 'train_config.dataset.repo_id'
                else:
                    row['pairing'] = f'policy-only: dataset.repo_id is {type(repo_id).__name__}: {str(repo_id)[:120]}'
            except Exception as exc:
                row['pairing'] = f'policy-only: train_config unreadable ({type(exc).__name__})'
        else:
            row['pairing'] = 'policy-only: no train_config.json'
        rows.append(row)
    for row in rows:
        row['set'] = 'dev' if row['rank'] < dev else 'heldout'
    selection = {'protocol': 'docs/real_world/CENSUS_PROTOCOL_A_20261022.md', 'listing_url': LISTING,
                 'listing_sha256': sha(listing), 'retrieved': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 'rows': rows, 'skipped': skipped}
    (out / 'selection.json').write_text(json.dumps(selection, indent=1) + '\n')
    print(f'selected {len(rows)} rows ({sum(r["set"] == "dev" for r in rows)} dev), skipped {len(skipped)}; '
          f'paired {sum(bool(r["dataset"]) for r in rows)}')


def extend(selection: Path, label: str, count: int) -> None:
    """Continue the retained listing after the last selected row (protocol A, amendment 2)."""
    document = json.loads(selection.read_text())
    entries = json.loads((selection.parent / 'listing.json').read_bytes())
    start = max(r['listing_index'] for r in document['rows']) + 1
    added, skipped = [], []
    for index in range(start, len(entries)):
        if len(added) >= count:
            break
        entry, model = entries[index], entries[index]['id']
        if entry.get('library_name') != 'lerobot':
            skipped.append({'model': model, 'reason': f"library_name is {entry.get('library_name')!r}"})
            continue
        time.sleep(0.3)
        try:
            info_bytes = get(f'{API}/models/{model}/revision/main')
            info = json.loads(info_bytes)
        except Exception as exc:
            skipped.append({'model': model, 'reason': f'revision API failed: {type(exc).__name__}: {exc}'})
            continue
        listed = {s.get('rfilename') for s in info.get('siblings', []) if isinstance(s, dict)}
        if info.get('private') is not False or info.get('gated') not in (None, False):
            skipped.append({'model': model, 'reason': 'private or gated'})
            continue
        if 'config.json' not in listed:
            skipped.append({'model': model, 'reason': 'no config.json'})
            continue
        commit = info['sha']
        row = {'rank': len(document['rows']) + len(added), 'listing_index': index, 'model': model, 'model_commit': commit,
               'downloads': entry.get('downloads'), 'likes': entry.get('likes'), 'revision_api_sha256': sha(info_bytes),
               'dataset': None, 'pairing': 'policy-only: no train_config.json', 'set': label}
        if 'train_config.json' in listed:
            try:
                raw = get(f'https://huggingface.co/{model}/resolve/{commit}/train_config.json', 2 * 1024 * 1024)
                repo_id = json.loads(raw).get('dataset', {}).get('repo_id')
                row['train_config_sha256'] = sha(raw)
                if isinstance(repo_id, str) and repo_id.count('/') == 1:
                    row['dataset'], row['pairing'] = repo_id, 'train_config.dataset.repo_id'
                else:
                    row['pairing'] = f'policy-only: dataset.repo_id is {type(repo_id).__name__}: {str(repo_id)[:120]}'
            except Exception as exc:
                row['pairing'] = f'policy-only: train_config unreadable ({type(exc).__name__})'
        added.append(row)
    document['rows'] += added
    document.setdefault('extensions', []).append({'set': label, 'retrieved': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                  'start_listing_index': start, 'rows': len(added), 'skipped': skipped})
    selection.write_text(json.dumps(document, indent=1) + '\n')
    print(f'extended with {len(added)} {label} rows from listing index {start}; skipped {len(skipped)}; '
          f'paired {sum(bool(r["dataset"]) for r in added)}')


def run(selection: Path, which: str, out: Path) -> None:
    from policy_doctor import __version__, hub

    urls = []
    original = hub._opener

    class Logged:
        def __init__(self, opener):
            self.opener = opener

        def open(self, request, timeout=None):
            urls.append(request.full_url)
            return self.opener.open(request, timeout=timeout)

    hub._opener = lambda: Logged(original())
    rows = [r for r in json.loads(selection.read_text())['rows'] if r['set'] == which]
    out.mkdir(parents=True, exist_ok=False)
    for row in rows:
        urls.clear()
        start = time.monotonic()
        record = {'row': row, 'product_version': __version__}
        try:
            report = hub.check_checkpoint(f'hf://{row["model"]}', f'hf://{row["dataset"]}' if row['dataset'] else None,
                                          fetch=True, revision=row['model_commit'])
            record.update(outcome='report', report=report)
        except Exception as exc:
            record.update(outcome='intake_error', error_type=type(exc).__name__, error=str(exc)[:1000],
                          traceback=traceback.format_exc()[-2000:])
        record['seconds'] = round(time.monotonic() - start, 3)
        record['requests'] = list(urls)
        name = f'{row["rank"]:03d}_{row["model"].replace("/", "__")}.json'
        (out / name).write_text(json.dumps(record, indent=1, default=str) + '\n')
        status = record.get('report', {}).get('status', record['outcome'])
        print(f'{row["rank"]:3d} {row["model"]:60s} {status:12s} {record["seconds"]:6.2f}s', flush=True)
        time.sleep(0.5)


UNMODELED = re.compile(r'Custom (pre|post)processor (\S+) is not validated')


def summary(directory: Path) -> dict:
    records = [json.loads(p.read_text()) for p in sorted(directory.glob('*.json'))]
    status, codes, unmodeled, intake = collections.Counter(), collections.Counter(), collections.Counter(), []
    attention, evaluated, paired = [], 0, 0
    for r in records:
        if r['outcome'] != 'report':
            status['intake_error'] += 1
            intake.append({'model': r['row']['model'], 'dataset': r['row']['dataset'], 'error': r['error']})
            continue
        report = r['report']
        status[report['status']] += 1
        paired += bool(r['row']['dataset'])
        found = {f['code'] for f in report['findings']}
        if r['row']['dataset'] and 'dataset.metadata_format_unsupported' not in found and 'hub.metadata_unavailable' not in found:
            evaluated += 1
        for f in report['findings']:
            codes[(f['code'], f['status'])] += 1
            m = UNMODELED.search(f.get('message', ''))
            if m:
                unmodeled[m.group(2)] += 1
            if f['status'] in ('error', 'warning'):
                attention.append({'model': r['row']['model'], 'dataset': r['row']['dataset'], 'code': f['code'],
                                  'status': f['status'], 'message': f['message'][:400],
                                  'evidence': [{k: e.get(k) for k in ('path', 'pointer', 'value')} for e in f.get('evidence', [])[:3]]})
    result = {'rows': len(records), 'status': dict(status), 'paired_reports': paired,
              'paired_with_dataset_comparison_evaluated': evaluated, 'intake_errors': intake,
              'unmodeled_steps': unmodeled.most_common(), 'finding_codes': sorted(([*k, v] for k, v in codes.items()), key=lambda x: -x[2]),
              'error_and_warning_findings': attention,
              'median_seconds': sorted(r['seconds'] for r in records)[len(records) // 2] if records else None}
    (directory.parent / f'{directory.name}_summary.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select')
    s.add_argument('out', type=Path)
    s.add_argument('--dev', type=int, default=60)
    s.add_argument('--heldout', type=int, default=40)
    r = sub.add_parser('run')
    r.add_argument('selection', type=Path)
    r.add_argument('set')
    r.add_argument('out', type=Path)
    e = sub.add_parser('extend')
    e.add_argument('selection', type=Path)
    e.add_argument('label')
    e.add_argument('--count', type=int, default=40)
    m = sub.add_parser('summary')
    m.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'select':
        select(args.out, args.dev, args.heldout)
    elif args.command == 'run':
        run(args.selection, args.set, args.out)
    elif args.command == 'extend':
        extend(args.selection, args.label, args.count)
    else:
        result = summary(args.directory)
        print(json.dumps({k: v for k, v in result.items() if k not in ('error_and_warning_findings', 'finding_codes')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
