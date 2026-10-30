# Monitor an explicit dataset inventory

Run one explicit dataset inventory repeatedly from cron or CI. Each cycle produces private health-card artifacts, durable run/history records and a local static HTML index. There is no hosted service, repository discovery, whole-Hub census, publication, message sending or automatic cron installation.

The monitor uses the existing [health-card engine](HEALTH_CARDS.md). A badge describes bounded checks and their coverage. A consumer name is an operator-supplied label; it does not prove that a native trainer has run. Neither a passing card nor a successful monitoring cycle guarantees whole-dataset quality, physical safety or task success.

## Start an inventory

Save `inventory.json`:

```json
{
  "schema_version": 1,
  "datasets": [
    {
      "id": "ur5-input",
      "source": "hf://lerobot/berkeley_autolab_ur5",
      "revision": "c4e26a697fc4c04776b0558f83e563d14be0109f",
      "consumer": "my-input-review",
      "config": {
        "max_bytes": 16777216,
        "max_rows": 200000,
        "video_headers": true
      }
    }
  ]
}
```

Use an immutable commit for reproducible checks, or `"revision": "main"` to monitor a moving public dataset head. Up to 16 explicit entries are accepted. IDs are distinct bounded ASCII names without paths. Each entry accepts only `id`, `source`, `revision`, `consumer`, `config`; config uses the health-card option allowlist and records every effective default. Tokens and arbitrary HTTP URLs are not inventory inputs.

```bash
roboclue advanced monitor scan-once \
  --inventory ./inventory.json --state ./private-monitor --fetch --timeout 300
```

The state path must be separate from local dataset/consumer inputs. A new state directory is private by default. Open `private-monitor/index.html` locally, or serve that directory through an operator-configured private static server. No web server is launched by the monitor. Review paths and configuration details before sharing artifacts.

Use the installed `roboclue` command from the environment where RoboClue is
installed. Native Parquet checks require the dataset dependencies.

## What happens every cycle

For a public-HF entry, the monitor first anonymously revalidates the exact repository identity, public/ungated status and requested revision through the existing bounded HTTPS intake. A mutable head resolves to one immutable commit. New assessments use that pinned commit. No credentials, SDK cache, weights or remote repository code are used.

A cache key includes the explicit source, resolved commit, consumer label, complete effective configuration, product source identities, Python runtime and relevant dependency versions. A cache hit still requires a successful public identity/revision check in the current cycle. It reuses the prior immutable assessment; it does not claim that rows were read again. The index labels this distinction.

Changed revision, engine, consumer or config selects a different key. Incomplete/unknown assessments are retried rather than cached. Recorded errors can be cached when their reported components completed. Local datasets are reassessed each cycle. Configurations naming a checkpoint, deployment file, GR00T modality file or OpenPI source directory also bypass cache: a path string cannot establish unchanged external input contents.

Offline HF cycles, private/gated repositories, identity mismatches, timeouts, invalid inputs and incomplete scans publish current unknown outcomes; historical passing cards never substitute for them. Local entries can run without `--fetch`. Supported data contradictions remain errors independently of another component's uncertainty.

Before work begins, the published index changes to **unknown / in progress**. A nonblocking process lock refuses overlapping cycles. Workers have separate process sessions and bounded result JSON; their stdout/stderr are discarded. On deadline, the monitor terminates then kills the worker's process group and reaps the direct worker. Ordinary descendants in that group are terminated; this is not a sandbox for hostile code that deliberately escapes its group. The engine's structured error type/message remains in the result.

## Artifacts and restart

```text
private-monitor/
  index.html                         current bounded outcomes and retained-history links
  state.json                         current cycle and last 64 history summaries
  monitor.lock                       nonblocking process lock
  cache/<key>.json                   verified immutable assessment envelopes
  runs/<run-id>/report.json          immutable cycle report, engine/config identities
  runs/<run-id>/<id>/monitor-result.json
  runs/<run-id>/<id>/badge.svg        current outcome badge
  runs/<run-id>/<id>/card/           health-card.json, badge.svg, summary.txt when assessed
```

Resolved/scan request and response JSON are retained with each entry. Unknown cards retain their actual unassessed coverage. The current index shows the latest cycle; its history links are explicitly historical. Saved files are finite static snapshots, not a promise of freshness between scheduled polls.

Files publish through temporary files, `fsync` and atomic replacement. Completed run directories are retained. Restart after an interrupted cycle preserves the incomplete directory and adds an `interrupted.json` marker before trying a new cycle. A process that stops mid-cycle leaves the in-progress unknown page rather than advancing a previous pass. Hashes detect accidental modification; they are not signatures or independent public-origin authentication.

State, inventory and cache inputs must be bounded regular files. Symlink/special-file state paths are refused, including FIFOs. Corrupt cache entries produce unknown rather than an unverified pass; remove or archive the affected entry before retrying. The monitor does not follow arbitrary inventory path IDs or authorize authenticated repositories.

## Scheduling, CI and retention

The module runs one cycle and exits; cron, a private job runner or CI owns recurrence. For example, install a cron entry yourself using absolute paths:

```cron
17 * * * * /path/to/env/bin/roboclue advanced monitor scan-once --inventory /path/to/inventory.json --state /path/to/private-monitor --fetch --timeout 300 > /path/to/monitor-last.json 2> /path/to/monitor-last.stderr
```

The two redirected files are overwritten each cycle; complete run evidence remains under state. CLI exits are 0 for pass/warning, 1 for a supported error, 3 for current unknown, 2 for invalid invocation/inventory, and 75 when another cycle owns the lock. A nonzero outcome still retains its cycle artifacts when state storage is usable.

Default operator limits are **128 retained runs, 256 cache entries, 512 MiB of run/cache/interrupted data**, with an 8192-file hard bound. `--max-runs`, `--max-cache-entries`, and `--max-data-mb` set explicit limits within the implementation caps. The byte check reserves the bounded next cycle before scanning; it can stop before actual bytes reach the selected cap. Existing cache hits work at the cache-entry limit; a new uncached assessment at that limit becomes unknown.

When a quota prevents scanning, the current index/state become unknown and `monitor-status.json` records the reason. No new assessment/card/cache is created and nothing historical is automatically deleted. The small current metadata files are separately bounded to 2 MiB each. Archive old run directories and unused cache files outside the active state, or start a new private state directory. Keep enough space for current metadata and temporary atomic writes. Hourly monitoring reaches the default run-count limit after roughly five days; choose and maintain a retention policy suited to the schedule. This is an explicit bounded private workflow, not an unattended hosting or storage-retention service.

## Python API

```python
from roboclue import scan_once

result = scan_once(
    "inventory.json", "private-monitor", fetch=True,
    timeout_seconds=300, max_runs=128,
    max_cache_entries=256, max_data_bytes=512 * 1024**2,
)
print(result["status"], result["run_id"])
```

The deadline is per dataset, shared by revision resolution and scanning, and can be at most 900 seconds, followed by bounded process-cleanup grace on cancellation. A valid inventory may therefore take up to 16 such bounded operations. Each cycle serializes its explicit inputs; an overlapping invocation exits promptly instead of duplicating capacity.

Validation includes actual local module-CLI artifacts, process-group descendant cancellation, cache/consumer/config/engine changes, offline/failure-after-pass behavior, quota controls, interrupted-run recovery, public identity/auth refusal and special-file controls. Generated checker envelopes in unit tests are controls, not public incidents or evidence of saved training cost. Selected live public cycle results are retained separately with the qualification receipts; no whole-Hub coverage or community adoption is implied.
