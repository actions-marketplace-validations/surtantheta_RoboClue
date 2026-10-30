# Check one artifact

```sh
roboclue check /path/to/artifact
```

The default answer has three lines: the scoped verdict, the main problem or
coverage gap, and a next command. `--json` prints the original machine report;
`--explain` appends detailed evidence and scope. Input errors with `--json`
also produce a JSON object on stdout and exit2. `--output NEW_FILE` saves the
original JSON report regardless of presentation. Existing files are refused.
Reports must be saved outside the inspected input directories.

| Artifact | Detection and assessment |
| --- | --- |
| Dataset directory or `meta/info.json` | Existing metadata, rows and saved-statistics self-consistency; generic health reads video headers only with `--video-headers` |
| Checkpoint directory or recognized feature configuration | Existing checkpoint metadata checks; no weights loaded, no inference |
| Saved native target capture JSON | Original verifier, reviewed `--reference` and independently retained `--pin-file` or `--criterion-sha256` |
| RoboClue flight recording JSONL | Original software event checks and missing-evidence findings; no hardware causal guarantee |
| Native upgrade-impact manifest JSON | Original retained criterion and asset comparisons |
| Other JSON configuration | Bounded strict parsing; with `--reference`, configuration value comparison; semantic correctness otherwise unknown |
| Isaac Lab / robomimic HDF5 | Simulator convention checks; `--converted` compares the source and candidate |
| `meta/modality.json` or explicit GR00T selectors | Version/config-specific contract; choose the intended config before considering dataset changes |

A directory containing both dataset and checkpoint metadata needs an explicit
`--kind dataset` or `--kind checkpoint`. Unsupported files are refused rather
than executed. A field containing `status: pass` is never trusted as a verdict.
Missing paths, malformed JSON, symlinks and special files are input errors.

## Ask a consumer-specific question

`--robot-command` or `--robot-config` runs the recorded-units comparison against
those declared settings. The command is parsed, never executed. `--units` alone
identifies dataset units and explicitly leaves deployment unassessed.
Unknown robot-option names are invalid input. Explicit camera declarations
also compare recorded/declared key identities: disjoint names without a declared
mapping are a key mismatch; partial or empty selections produce a warning requiring review because
a future policy may intentionally use fewer views. Image sizes, rates, joint
order and policy-view requirements are separate. See [robot declarations](ROBOT_DECLARATIONS.md).

Unset robot-unit settings use the qualified source profile selected by the actual
declared version. Without that version the canonical consumer comparison stays
incomplete. RoboClue cannot infer the robot's runtime version from its own Python
environment. In the tested LeRobot sources,
`lerobot-record` is collection-only; policy execution uses the native
rollout entry point. This check parses settings from either command and does
not execute it or validate its runtime role.

Add `--lerobot-version YOUR_ACTUAL_VERSION` alongside the actual command/config.
Tested units/default profiles cover published 0.6.1 and recorded 0.6.2 sources for
SO-100/101, Koch, OMX, bimanual SO and the LeKiwi arm. This does not attest installed
bytes or qualify every build with a matching label. Missing or unqualified labels leave the consumer
comparison UNKNOWN, including required CI gates. The recorded-unit inference is
retained. The tool's local framework metadata is reported separately.

`--groot-version`, `--groot-config` and `--groot-config-name` select the actual
GR00T metadata/config contract. Config Python is parsed, never executed. A
consumer mismatch is a reason to review the selected config; it is not an
instruction to change a public dataset merely to silence a finding.

`--openpi-config` plus `--openpi-source-root` reaches the existing convention
check. “Delta-transform choice: consistent” covers that choice only. A separate
raw-input check compares the selected adapter's declared keys, and reports the
dataset format alongside OpenPI's verified reader lockfile pin. Matching keys do
not establish installed-loader compatibility, image contents or native training.
Convention warnings/errors remain visible beside a consistent delta choice;
`--fail-on warning` fails on a warning in that routed component even if unrelated
video coverage is unknown. A required convention assessment with undecided
coordinates names them under `coverage.required_assessments`; a passed delta
choice does not establish those unknowns.
The full report retains its overall status. `selected_assessment` identifies
the convention component used for the scoped question; an unrelated video gap
does not hide its warning. Errors elsewhere still block the command. Severity
thresholds include component statuses even when finding examples were bounded.

`--question recovery` runs the recording ledger check explicitly. Observed
truncated files or destructive resume hazards also select it automatically.
Missing videos alone do not establish an interrupted recording. Plans describe
a separate candidate copy; original-directory resume approval and successful
native reloading require their own evidence.
Explicit recovery reads bounded MP4 headers by default, matching the specialist
checker and resume gate. `--no-video-headers` opts out and leaves the resulting
coverage gap explicit; the next command asks for headers when needed.

To apply a supported plan explicitly, use
`roboclue check DATASET --write-copy NEWDIR`. The destination must be new and
outside the source. The operation recomputes the plan, copies supported files
and rewrites into that separate directory, preserves the original, rebuilds
supported statistics from retained episode summaries and checks the result.
An unestablished statistics or integrity result stays UNKNOWN; creating a copy
does not establish native loading, actual resume or training. See
[recording recovery](RECORDING_SALVAGE.md) for bounds and supported operations.

## Require a comparison in CI

```sh
roboclue check DATASET --robot-command "$DEPLOY_COMMAND" \
  --lerobot-version "$LEROBOT_VERSION" --require deployment-units --json
```

This requires an actual declared consumer. A missing command, unrecognized robot
type/units flag, non-dataset path or unresolved required comparison cannot clear
the gate. Unknown required coverage exits 3 even with `--fail-on error`.
An error-only threshold without `--require` is only a severity filter; it does
not prove every possible check ran. Missing unrelated video evidence does not
fail the scoped units comparison.

## Missing evidence is useful information

`INCOMPLETE` presents the original `unknown` status in plain words. It does not
change a raw verdict or the exit code. A dataset can have passing metadata and
numeric checks while videos or statistics are missing. This does not establish
a defect, and it does not approve the data for training.

`--video-headers` checks available local headers or bounded public range reads;
it does not decode images or download complete video files. RoboClue does not
invent a `--fetch-videos` flag. If files or statistics are absent, restore the
actual source-backed files or generate statistics through your dataset's native
workflow; changing the schema merely to silence a diagnostic is not a repair.
`--explain` contains the precise missing component and source witnesses.
Machine `files_missing` counts files observed missing among examined references;
zero alongside unread/unassessed files does not establish that every video is
present. Saved-statistics consistency similarly does not compare their values
with dataset rows or prove that the data describe a robot.

## Public datasets are explicit downloads

```sh
roboclue check hf://lerobot/berkeley_autolab_ur5 \
  --fetch --revision c4e26a697fc4c04776b0558f83e563d14be0109f --video-headers
```

Hub inputs default to datasets. For a model, pass `--kind checkpoint`.
No network intake occurs without `--fetch`. Reads are bounded by `--max-bytes`,
`--max-rows` and `--max-files`; the default dataset byte budget is 64 MiB. Public
scans can take minutes. They are not the first-use demo or a whole-Hub audit.
For your own first check, substitute `hf://OWNER/DATASET --fetch`; the offline
synthetic demo is a separate installation/control example. The intake does not
use a persistent Hub SDK cache: repeating the command fetches the bounded public
inputs again. Use an existing local dataset path for offline repeated checks.
A dataset containing `meta/modality.json` reaches the GR00T profiles; this does
not select an intended consumer version. Use `--groot-version n1.5` or `n1.7`
when that is your actual consumer. `--explain` preserves the exact
selected contract and any remaining coverage limits.

## Replay a saved target capture

```sh
roboclue check ./roboclue-demo/broken.json \
  --reference ./roboclue-demo/reference-contract.json \
  --pin-file ./roboclue-demo/criterion.sha256
```

The demo's fault exits 1. Substitute `corrected.json` to see the covered check
pass (exit 0), or `unknown.json` to see missing source coverage (exit 3). Keep the
same reference and pin for each. This is a synthetic onboarding criterion,
separate from the public native trainer proof and from your own intended recipe.

## Automation

| Exit | Meaning |
| --- | --- |
| 0 | Selected covered checks pass or compared values match; this is not whole-pipeline certification |
| 1 | Supported error, changed values, or a warning requiring review |
| 2 | Invalid, missing, ambiguous or unsupported input |
| 3 | Incomplete assessment / missing evidence (`unknown`) |

For automation, use `--json` and check its original `status`, not only exit 0.
Configuration comparisons retain the existing `equal` Boolean and `changes`
list instead of adding a `status`; equality exits 0 and differences exit 1.
A warning is not a certification. `--explain` and `--json` preserve evidence
hashes, original source identities and scope. Rendering never rewrites health
card IDs or the target criterion.

The default canonical command does not clear warnings. An explicit
`--fail-on error` is an advisory severity filter and can return zero for a warning;
it still cannot waive incomplete evidence for a required assessment.
