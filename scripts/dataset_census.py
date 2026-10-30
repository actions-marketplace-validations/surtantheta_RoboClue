#!/usr/bin/env python3
"""Census protocol B: RoboClue dataset checks versus lerobot-doctor on the same public LeRobot v3 bytes.

select   : frozen selection rule (docs/real_world/CENSUS_PROTOCOL_B_20261022.md)
fetch    : download meta/ and data/ at the pinned commit (no videos), verifying listed sizes
run      : RoboClue scan + window audit + dataset summary, and lerobot-doctor, on each local copy
summary  : aggregate one run directory
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

API = 'https://huggingface.co/api'
LISTING = f'{API}/datasets?filter=LeRobot&sort=downloads&direction=-1&limit=500'
# lerobot-doctor 0.2.0 registry minus `videos` (videos are not downloaded).
DOCTOR_CHECKS = 'metadata,temporal,actions,statistics,episodes,consistency,training,anomalies,portability,per_episode'


def get(url: str, limit: int = 16 * 1024 * 1024) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': 'RoboClue-census/1'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError(f'response over {limit} bytes: {url}')
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise AssertionError


def select(out: Path, dev: int, heldout: int) -> None:
    out.mkdir(parents=True, exist_ok=False)
    listing = get(LISTING)
    (out / 'listing.json').write_bytes(listing)
    entries = json.loads(listing)
    tagged = sum('LeRobot' in (e.get('tags') or []) for e in entries)
    if not entries or tagged < 0.8 * len(entries):
        raise SystemExit(f'Listing is not a LeRobot listing ({tagged}/{len(entries)}); refusing to select')
    rows, skipped = _eligible(entries, 0, dev + heldout)
    for row in rows:
        row['set'] = 'dev' if row['rank'] < dev else 'heldout'
    (out / 'selection.json').write_text(json.dumps({
        'protocol': 'docs/real_world/CENSUS_PROTOCOL_B_20261022.md', 'listing_url': LISTING,
        'listing_sha256': hashlib.sha256(listing).hexdigest(), 'retrieved': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'rows': rows, 'skipped': skipped}, indent=1) + '\n')
    print(f'selected {len(rows)} ({sum(r["set"] == "dev" for r in rows)} dev), skipped {len(skipped)}')


def extend(selection: Path, label: str, count: int) -> None:
    """Continue the retained listing after the last selected row (protocol B, amendment 2)."""
    document = json.loads(selection.read_text())
    entries = json.loads((selection.parent / 'listing.json').read_bytes())
    start = max(r['listing_index'] for r in document['rows']) + 1
    rows, skipped = _eligible(entries, start, count, first_rank=len(document['rows']))
    for row in rows:
        row['set'] = label
    document['rows'] += rows
    document.setdefault('extensions', []).append({'set': label, 'retrieved': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                  'start_listing_index': start, 'rows': len(rows), 'skipped': skipped})
    selection.write_text(json.dumps(document, indent=1) + '\n')
    print(f'extended with {len(rows)} {label} rows from listing index {start}; skipped {len(skipped)}')


def _eligible(entries, start, count, first_rank=0):
    rows, skipped = [], []
    for index in range(start, len(entries)):
        entry = entries[index]
        if len(rows) >= count:
            break
        name = entry['id']
        time.sleep(0.3)
        try:
            info = json.loads(get(f'{API}/datasets/{name}/revision/main?blobs=true'))
        except Exception as exc:
            skipped.append({'dataset': name, 'reason': f'revision API: {type(exc).__name__}'})
            continue
        if info.get('private') is not False or info.get('gated') not in (None, False):
            skipped.append({'dataset': name, 'reason': 'private or gated'})
            continue
        files = {s['rfilename']: s.get('size') for s in info.get('siblings', []) if isinstance(s, dict) and 'rfilename' in s}
        if 'meta/info.json' not in files:
            skipped.append({'dataset': name, 'reason': 'no meta/info.json'})
            continue
        commit = info['sha']
        try:
            meta = json.loads(get(f'https://huggingface.co/datasets/{name}/resolve/{commit}/meta/info.json', 4 * 1024 * 1024))
        except Exception as exc:
            skipped.append({'dataset': name, 'reason': f'info.json unreadable: {type(exc).__name__}'})
            continue
        selected = {k: v for k, v in files.items() if k.startswith(('meta/', 'data/'))}
        size = sum(v or 0 for v in selected.values())
        if meta.get('codebase_version') != 'v3.0':
            skipped.append({'dataset': name, 'reason': f"codebase_version {meta.get('codebase_version')!r}"})
            continue
        if not isinstance(meta.get('total_frames'), int) or meta['total_frames'] > 200000:
            skipped.append({'dataset': name, 'reason': f"total_frames {meta.get('total_frames')!r} over 200000"})
            continue
        if any(v is None for v in selected.values()) or size > 300 * 1024 * 1024:
            skipped.append({'dataset': name, 'reason': f'meta+data {size} bytes (limit 300 MB) or unknown sizes'})
            continue
        rows.append({'rank': first_rank + len(rows), 'listing_index': index, 'dataset': name, 'commit': commit,
                     'downloads': entry.get('downloads'), 'total_frames': meta['total_frames'],
                     'total_episodes': meta.get('total_episodes'), 'robot_type': meta.get('robot_type'),
                     'files': selected, 'bytes': size})
    return rows, skipped


def fetch(selection: Path, which: str, scratch: Path) -> None:
    rows = [r for r in json.loads(selection.read_text())['rows'] if r['set'] == which]
    for row in rows:
        root = scratch / f'{row["rank"]:03d}_{row["dataset"].replace("/", "__")}'
        manifest = {}
        for name, size in sorted(row['files'].items()):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://huggingface.co/datasets/{row["dataset"]}/resolve/{row["commit"]}/{name}'
            data = get(url, limit=size + 1)
            if len(data) != size:
                raise SystemExit(f'size mismatch {row["dataset"]}/{name}: {len(data)} != {size}')
            target.write_bytes(data)
            manifest[name] = hashlib.sha256(data).hexdigest()
        (root.parent / (root.name + '.manifest.json')).write_text(json.dumps(manifest, indent=1))
        print(f'fetched {row["dataset"]} ({row["bytes"]} bytes, {len(manifest)} files)', flush=True)


def run(selection: Path, which: str, scratch: Path, doctor: Path, out: Path) -> None:
    from policy_doctor import __version__
    from policy_doctor.artifacts import load_dataset
    from policy_doctor.dataset import scan_dataset
    from policy_doctor.windows import audit_episode_windows

    rows = [r for r in json.loads(selection.read_text())['rows'] if r['set'] == which]
    out.mkdir(parents=True, exist_ok=False)
    for row in rows:
        root = scratch / f'{row["rank"]:03d}_{row["dataset"].replace("/", "__")}'
        record = {'row': {k: v for k, v in row.items() if k != 'files'}, 'product_version': __version__}
        frames = max(1, min(row['total_frames'], 1_000_000))
        for label, call in (('summary', lambda root=root: load_dataset(str(root)).summary()),
                            ('scan', lambda root=root, frames=frames: scan_dataset(str(root), max_rows=frames, max_files=1000)),
                            ('windows', lambda root=root: audit_episode_windows(str(root), max_rows=200000, max_files=128))):
            start = time.monotonic()
            try:
                record[label] = {'outcome': 'report', 'report': call()}
            except Exception as exc:
                record[label] = {'outcome': 'intake_error', 'error_type': type(exc).__name__, 'error': str(exc)[:1000],
                                 'traceback': traceback.format_exc()[-1500:]}
            record[label]['seconds'] = round(time.monotonic() - start, 3)
        start = time.monotonic()
        try:
            proc = subprocess.run([str(doctor), 'check', str(root), '--json', '-v', '--checks', DOCTOR_CHECKS],
                                  capture_output=True, text=True, timeout=900)
            try:
                body = json.loads(proc.stdout)
            except ValueError:
                body = None
            record['lerobot_doctor'] = {'exit': proc.returncode, 'report': body,
                                        'stdout_head': None if body is not None else proc.stdout[:2000],
                                        'stderr_tail': proc.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            record['lerobot_doctor'] = {'exit': None, 'error': 'timeout 900 s'}
        record['lerobot_doctor']['seconds'] = round(time.monotonic() - start, 3)
        (out / f'{root.name}.json').write_text(json.dumps(record, indent=1, default=str) + '\n')
        rc = {k: (record[k]['report'].get('status') if record[k]['outcome'] == 'report' and isinstance(record[k]['report'], dict) else record[k]['outcome'])
              for k in ('scan', 'windows')}
        print(f'{row["rank"]:3d} {row["dataset"]:55s} scan={rc["scan"]} windows={rc["windows"]} '
              f'doctor={(record["lerobot_doctor"].get("report") or {}).get("overall_severity")} exit={record["lerobot_doctor"].get("exit")}', flush=True)


def summary(directory: Path) -> dict:
    records = [json.loads(p.read_text()) for p in sorted(directory.glob('*.json'))]
    result = {'rows': len(records), 'roboclue': collections.defaultdict(collections.Counter),
              'doctor_overall': collections.Counter(), 'doctor_fail_messages': [], 'roboclue_attention': [],
              'intake_errors': []}
    for r in records:
        name = r['row']['dataset']
        for label in ('scan', 'windows'):
            part = r[label]
            if part['outcome'] != 'report':
                result['roboclue'][label]['intake_error'] += 1
                result['intake_errors'].append({'dataset': name, 'tool': label, 'error': part['error']})
                continue
            report = part['report']
            result['roboclue'][label][report.get('status')] += 1
            for f in report.get('findings', []):
                if f.get('status') in ('error', 'warning'):
                    result['roboclue_attention'].append({'dataset': name, 'tool': label, 'code': f.get('code'),
                        'status': f.get('status'), 'message': str(f.get('message'))[:400],
                        'evidence': [{k: e.get(k) for k in ('path', 'pointer', 'value')} for e in f.get('evidence', [])[:2]]})
        doc = r['lerobot_doctor'].get('report') or {}
        result['doctor_overall'][doc.get('overall_severity', f"exit {r['lerobot_doctor'].get('exit')}")] += 1
        for check in doc.get('checks', []) if isinstance(doc.get('checks'), list) else []:
            for m in check.get('messages', []):
                if str(m.get('severity', check.get('severity', ''))).upper() == 'FAIL':
                    result['doctor_fail_messages'].append({'dataset': name, 'check': check.get('name'),
                                                           'message': str(m.get('message'))[:400]})
    result['roboclue'] = {k: dict(v) for k, v in result['roboclue'].items()}
    (directory.parent / f'{directory.name}_summary.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select')
    s.add_argument('out', type=Path)
    s.add_argument('--dev', type=int, default=20)
    s.add_argument('--heldout', type=int, default=10)
    f = sub.add_parser('fetch')
    f.add_argument('selection', type=Path)
    f.add_argument('set')
    f.add_argument('scratch', type=Path)
    r = sub.add_parser('run')
    r.add_argument('selection', type=Path)
    r.add_argument('set')
    r.add_argument('scratch', type=Path)
    r.add_argument('doctor', type=Path)
    r.add_argument('out', type=Path)
    e = sub.add_parser('extend')
    e.add_argument('selection', type=Path)
    e.add_argument('label')
    e.add_argument('--count', type=int, default=10)
    m = sub.add_parser('summary')
    m.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'select':
        select(args.out, args.dev, args.heldout)
    elif args.command == 'extend':
        extend(args.selection, args.label, args.count)
    elif args.command == 'fetch':
        fetch(args.selection, args.set, args.scratch)
    elif args.command == 'run':
        run(args.selection, args.set, args.scratch, args.doctor, args.out)
    else:
        res = summary(args.directory)
        print(json.dumps({k: v for k, v in res.items() if k not in ('doctor_fail_messages', 'roboclue_attention')}, indent=1))
        print('doctor FAIL messages:', len(res['doctor_fail_messages']), '| RoboClue error/warning findings:', len(res['roboclue_attention']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
