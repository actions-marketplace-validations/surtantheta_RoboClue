# CLI and MCP evidence interfaces

Use `roboclue check` for the main supported questions. The specialist commands
below retain the complete interfaces to four existing evidence APIs. They inspect
saved artifacts; they do not run a policy, send robot commands, train a model,
infer a migration rule or publish alerts. Installed CLI/MCP integration has separate package checks; these examples describe the interface contract.

The recurring workflows live under the same command's advanced group:

```sh
roboclue advanced monitor scan-once --inventory inventory.json --state ./private-health --fetch
roboclue advanced upgrade-watch scan-once --config watch.json --state-dir ./private-upgrades --json
roboclue advanced upgrade-watch catalog
```

See [health monitoring](HEALTH_MONITOR.md) and [owned-asset upgrade
watching](UPGRADE_WATCH.md) for their explicit inventories, scheduling, current
versus historical results, network and retention settings. These CLI operations
write their requested local state/artifacts; the MCP tools remain read-only.

## Dataset health card

```bash
roboclue advanced health-card /data/my-dataset --consumer my-training-job \
  --output /reports/new-card.json --artifacts /reports/new-card-artifacts

roboclue advanced health-card hf://ORG/DATASET --fetch --revision COMMIT_SHA \
  --max-bytes 67108864 --output /reports/new-public-card.json --json
```

`--output` writes a new complete JSON report. `--artifacts` explicitly requests a
new directory containing `health-card.json`, `badge.svg` and `summary.txt`. No
artifact directory is created unless requested. The JSON card, including its
configuration, observed revision, consumer label, coverage and content binding,
is retained unchanged; the consumer label does not establish native compatibility.

Use `--max-rows`, `--max-files` and `--max-bytes` for positive bounds. Boolean
options include `--video-headers`/`--no-video-headers`, `--units`/`--no-units`,
`--commanded-reach`/`--no-commanded-reach`, and
`--groot-contract`/`--no-groot-contract`. Missing units defaults to the existing
engine's automatic behavior. Local CLI users may pass `--config OPTIONS.json`
for the health-card engine's existing allowlisted options, including explicit
local consumer configuration paths. CLI flags override corresponding config
values. Invalid Boolean strings and numeric Booleans are refused.

Missing or partial evidence remains UNKNOWN. A known numeric error still remains
ERROR when a different component is partial. The badge describes only reported
bounded checks; it does not certify dataset quality, safety or training success.

## Flight-recording triage

```bash
roboclue advanced flight-triage /logs/completed-flight.jsonl \
  --max-command-gap-s 0.10 --max-feedback-gap-s 0.10 \
  --max-camera-gap-s 0.10 --max-latency-s 0.05 \
  --tracking-tolerance /criteria/joint-tolerances.json \
  --output /reports/new-flight-report.json --json
```

An example tolerance document is:

```json
{"elbow": {"unit": "rad", "absolute": 0.02}}
```

Create recordings with the opt-in Python API described in
[FLIGHT_RECORDER.md](FLIGHT_RECORDER.md). This command reads a completed recording;
it does not attach to a live producer. Findings name exact event IDs and remedies.
Latency requires explicitly associated feedback and aligned declared clock epochs.
Residual checks also require matching names, units and quantities. Missing
feedback or unsynchronized clocks stay UNKNOWN; a timestamp difference does not
establish motor delivery or a causal mechanical explanation.

## Native upgrade impact

```bash
roboclue advanced upgrade-impact /criteria/upgrade-manifest.json \
  --output /reports/new-impact.json --artifacts /reports/new-upgrade-alerts
```

The existing `native_upgrade_impact` manifest names a retained contract plus
reference/candidate source files, actual configurations and native captures for
each asset. Each file entry supplies `path` and `sha256`; relative paths resolve
beside the manifest. See the schema and coverage pins in `upgrade_impact.py`.
Version names alone do not establish compatibility. Unsupported native coverage
remains UNKNOWN. A changed configuration is an associated comparison, not proof
that this individual edit caused an observed boundary difference.

The optional artifact directory contains `impact.json` and `alerts.md`. Alerts
are local review files; no message, email, issue or workflow is sent. Reports
retain asset identity/status, coverage, reference/candidate checks, configuration
comparisons and the first observed source mismatch under the same retained rule.
Full and concise reports preserve `criterion_intent` and `comparison_basis`,
including whether a reference snapshot was explicitly adapted to the caller's
intended convention. A violated all-XYZW objective can coexist with unchanged raw
end-effector values in a legal field-specific convention. A `changed` result is a
retained-criterion violation; it does not establish an upstream bug or raw
old-to-new value change. Keep the original criterion and assess a different
legal objective under its own independently frozen contract.

## Offline native training-target check

```bash
roboclue advanced check-training-targets --contract /criteria/retained-contract.json \
  --criterion-sha256 INDEPENDENTLY_RETAINED_SHA256 \
  --capture /captures/actual-native-targets.json \
  --output /reports/new-target-check.json --json
```

Use the independently retained pin for the existing reference contract. Do not
rebuild it from the candidate being checked. The report preserves the first
source key, coordinate, expected/observed values or schema and retained provenance.
Uncovered source identities remain UNKNOWN. Captures do not authenticate their
producer. Live collator protection remains the installed Python API; this front
door only checks an explicit saved capture against a contract and pin.

Each retained bank requires the NumPy version recorded by its producer. The
qualified native demonstrations use these runtime profiles:

| Demonstration | Producer NumPy | Native consumer boundary |
| --- | --- | --- |
| GR00T N1.7 | 1.26.4 | Native trainer collation and actual loss input |
| LeRobot ACT | 2.2.6 | Actual policy input and original native update function |
| OpenPI PyTorch | 1.26.4 | Native loader and model-input hook with a controlled Torch consumer; Pi0 execution unassessed |

Run the saved-capture check or MCP server in the matching training environment.
The refusal message names the required and current versions. A newly declared
reference binds its own intended runtime; preserve that bank and pin through
the fault, correction and replay.

A LeRobot contract can also retain deterministic state/camera witnesses and
caller declarations. This saved target-capture format contains no observations
of those extra inputs. Such reports explicitly return
`additional_model_input_coverage.status="unknown"`, including in concise MCP
responses; their target/mask verdict does not certify the extra witnesses.

## CLI output and exit status

All four commands accept `--json` to print the full unmodified report. Without
it, they print an actionable summary, the most severe available findings,
coverage-bound limits and requested output paths. `--output NEW.json` is optional
and always denotes a new JSON file. Existing files, symlinks, FIFOs and missing
parents are refused. Artifact directories also must be new and have an existing
parent; use a separate destination from `--output`.

| Exit | Meaning |
| --- | --- |
| 0 | Scoped pass/healthy result, or advisory warning |
| 1 | Supported error or observed retained-boundary criterion violation (`changed`) |
| 2 | Invalid/unreadable input, refused destination or authorization failure; no assessment result |
| 3 | Missing/unsupported evidence (`unknown`) |

Reports and optional artifacts may describe failures; a nonzero assessment exit
does not mean the evidence report was not written. A destination error can leave
an earlier explicitly requested report file written; outputs are not an atomic
multi-file transaction. No existing output is overwritten.

## Read-only MCP registration

The host server calls:

```python
from roboclue import register_mcp_tools

register_mcp_tools(
    server, guard=guard, allowed=allowed_roots, annotations=read_only_annotations,
    allow_hub_datasets=False,
)
```

The tools are `dataset_health_card`, `triage_flight_recording`,
`assess_upgrade_impact` and `check_training_targets`. Every tool is read-only;
none accepts output paths, additional read roots, arbitrary health configuration,
training code or executable paths. The training tool takes only an explicit
contract, independent pin and capture, plus response format.

Health MCP exposes source/consumer/revision, explicit fetch, and bounded scalar
options only: `video_headers`, `max_rows`, `max_files`, `max_bytes`, `units`,
`commanded_reach` and `groot_contract`. It does not pass arbitrary nested
configuration, checkpoint/deployment paths or robot commands to preflight.
Public access requires **both** the server's `allow_hub_datasets=True` and the
request's `fetch=True` with an explicit `hf://` source. Local requests stay offline.
The tool advertises open-world access only when the operator enables that option.

Local dataset aliases are expanded to their effective dataset directory and
guarded before checker reads. Every entry and local link target in its bounded
directory inventory is authorized first; depth above 32 or more than 20,000
entries requires selecting a smaller explicit dataset. Special files are refused
through the host guard. The existing dataset engines retain their own containment
and file-integrity checks.

Upgrade authorization reads only the already-authorized bounded manifest to
discover its references. It guards the contract and every source/config/capture
from both sides of every asset before calling the assessment. Relative references
remain relative to the manifest; an outside target, including a link resolving
outside the roots, is refused. The assessment retains its file/hash postguards,
and the front door verifies the manifest document again after assessment.
Training checks authorize both paths before reading either; flight triage
authorizes the recording before reading it. These tools rely on the configured
host guard; they do not expand its roots.

Requests default to `response_format="concise"`. Concise JSON is bounded to 32
KiB, preserving original kind/status, counts, component coverage, named asset
verdicts, the first source mismatch and explicit omissions. Long text is withheld
with its hash rather than exposing a truncated remedy as an instruction. A
summary carries the full report hash. Detailed responses return the unchanged
report up to 192 KiB and otherwise return a bounded summary. Repeating a detailed
request rereads the artifacts; it is not a frozen report cache. For complete local
evidence and SVG/Markdown artifacts, run the named CLI with a new output path.
Concise health/upgrade replies include `local_cli_argv` with the actual authorized
source/manifest and selected options. Replace the explicitly marked output
placeholders with new local paths. This argument list is returned as data and is
never executed by MCP.
At the concise size bound, repeated per-asset intent/basis objects may use an
explicit `same_as_report` JSON pointer to the identical top-level object.

## Validation boundary

The front-door tests run actual health-card, flight-triage, upgrade-impact and
target-capture engines through CLI calls and the real FastMCP async API. They
exercise healthy, known-error/criterion-violation, corrected and UNKNOWN controls;
outside-root references and aliases; strict Boolean/bound handling; artifact
opt-in; and first-source witnesses. These are software and saved-artifact tests.
They do not qualify physical robot behavior. Installed alias/stdio integration was checked separately from these component controls; see [evidence](EVIDENCE.md).
