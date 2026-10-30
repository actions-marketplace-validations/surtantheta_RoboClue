# Supported workflows

Every finding covers its named inputs, framework profile and reading bounds.
Use [scope](SCOPE.md) to distinguish a completed check from unassessed behavior.
First choose [what your existing evidence can answer](CHOOSE_A_WORKFLOW.md).
File/configuration checks can help immediately; target correctness against intent
requires a reviewed retained reference, and physical questions need physical evidence.

## Dataset health

Inspect supported metadata, numeric rows and episode/window membership. Generic
health cards omit the GR00T dataset contract. `--consumer` is a freeform label;
`--groot-contract`, MCP `groot_contract=true`, or explicit GR00T recipe/version
configuration enables that contract. Cards retain effective configuration, coverage and source
identity. Monitoring maintains current results and history for an explicit local
or public-Hub inventory; your scheduler owns recurrence.

```sh
roboclue advanced health-card ./dataset --output ./health.json --artifacts ./health-card
roboclue advanced monitor scan-once --inventory inventory.json --state ./dataset-monitor --fetch
```

[Cards](HEALTH_CARDS.md) · [Monitoring](HEALTH_MONITOR.md) · [Preflight](PREFLIGHT.md)

A card does not certify general dataset health or native compatibility.
`roboclue check` routes units, GR00T configuration, simulator and recording-recovery
questions to the existing checks. The complete preflight options and specialist
workflows remain under `roboclue advanced`.

## Training-input preservation

Retain intended targets before a change, attach the supported native adapter,
and check the actual covered consumer boundary. Both contradictions and unknown
startup sources block consumption. Correct ordinary framework configuration and
rerun with the original reference/pin.

[GR00T](GR00T_TRAINING_GUARD.md) · [LeRobot](LEROBOT_TRAINING_GUARD.md) ·
[OpenPI](OPENPI_TRAINING_GUARD.md)

Saved captures can also be checked outside training:

```sh
roboclue advanced check-training-targets --contract startup-reference.json \
  --criterion-sha256 INDEPENDENTLY_RETAINED_SHA256 --capture native-capture.json --json
```

Replace the pin with the value retained from the intended reference. Saved
target captures do not include live camera/state witnesses; their extra-input
coverage remains UNKNOWN.

## Runtime observations

`RuntimeRecorder` wraps existing synchronous `send_action`/`get_observation`
software calls without replacing their behavior. Triage uses explicitly declared
criteria and cites recorded events. Request/return discrepancies do not establish
physical motor delivery.

```sh
roboclue advanced flight-triage ./runtime.jsonl --max-command-gap-s 0.10 --json
```

[Recorder integration](RUNTIME_RECORDER.md) · [Event schema](FLIGHT_RECORDER.md)

## Upgrade monitoring

Bind owned artifacts to saved version declarations, observe upstream changes,
and assess supported reference/candidate evidence against a retained criterion.
Version differences alone remain advisory/UNKNOWN. Alerts are local files.

```sh
roboclue advanced upgrade-watch catalog
roboclue advanced upgrade-watch scan-once --config watch.json --state-dir ./upgrade-monitor --json
```

[Owned assets](UPGRADE_WATCH.md) · [Impact evidence](UPGRADE_IMPACT.md)

## Conversion and recording workflows

| Job | Entry point | Scope |
| --- | --- | --- |
| Recover an interrupted LeRobot v3 recording | `roboclue advanced check-salvage` | Verified episode prefixes and a plan for a separate copy; no automatic application or footer-less recovery. [Guide](RECORDING_SALVAGE.md) |
| Check a supported source/export recipe | `roboclue advanced preflight` with original/recipe inputs | Numeric source associations for the supported Arena G1 recipe; arbitrary recipes are not inferred. [Criteria](CONVERSION_REFERENCE.md) |
| Compare reference/candidate consumer values | `roboclue advanced preflight` with explicit reference folders | Selected N1.7 G1 numeric windows; equality only when explicitly required. [Guide](CONSUMER_SAMPLE_COMPARISON.md) |
| Check simulator declarations | `roboclue advanced check-sim-data` | Supported Isaac Lab/robomimic conventions; inference assumptions remain visible. [Guide](SIM_DATA_PROVENANCE.md) |
| Inspect action/units declarations | Supported preflight configuration | Declared conflicts and bounded inference, not physical calibration. [Actions](ACTION_CONVENTIONS.md), [units](UNITS_MANIFEST.md) |

See `roboclue advanced --help` and the detailed guides for required arguments. None
of these routes establishes whole-dataset quality or task success.
