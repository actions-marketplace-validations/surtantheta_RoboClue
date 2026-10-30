# Dataset preflight

Start with `roboclue check DATASET`. Select the question when needed, such as
`--robot-command 'lerobot-record --robot.type=so101_follower ...'` for deployment
units, or `--groot-version n1.7 --groot-config CONFIG.py` for a GR00T recipe.
`DATASET` is an explicit local native LeRobot dataset root or a public
`hf://ORG/REPO` ID with `--fetch`.

The full specialist interface below is `roboclue advanced preflight`. It keeps
the existing detailed options and reports; the compact `check` answer focuses
on the selected question rather than presenting every unrelated readiness row.

For a required deployment-units gate, use:

```sh
roboclue check ./dataset \
  --robot-command 'lerobot-record --robot.type=so101_follower ...' \
  --require deployment-units --fail-on error --json
```

An unknown or missing required result fails this gate. A dataset-statistics pass
alone is never evidence that the native dataset loader or deployment can run.
The full interface also accepts `roboclue advanced preflight DATASET
--robot-command 'YOUR_COMMAND' --require deployment-units --fail-on error`.
It retains the complete preflight report and names the required comparison in
`coverage.required_assessments.deployment-units`; it does not overwrite the
overall status when unrelated video coverage remains incomplete.

## Saved statistics

The statistics component checks the supplied summaries' shape, finiteness and
internal consistency. A mean outside its saved min/max, or exactly zero standard
deviation with a meaningful nonzero saved range, is an ERROR under the same-cohort
interpretation, allowing `1e-6` relative plus `1e-8` absolute tolerance with a
scale floor of 1. These checks do not compare statistics with observed rows.
Zero spread can be legitimate for an unused dimension; a small positive spread
is not a constant actuator. Mode/epsilon and physical consequences remain
separate questions.

Dimension messages preview eight indices and explicitly mark omissions. Raw
`indices` preview at most sixteen, with `indices_total` and `indices_omitted`.
An unlisted index is not evidence that the remaining dimensions passed.

The command runs independent checks and produces one report. ERROR means at least one supported contradiction was found. UNKNOWN means some check that ran could not complete, whether you requested it or it ran automatically (the GR00T dataset contract runs for a local dataset with `meta/modality.json`, and the units manifest with a robot command or config). WARNING requires review for the intended consumer. PASS means the reported checks completed within their coverage. An error stays visible when another check is incomplete.

## Public datasets

```sh
roboclue advanced preflight hf://lerobot/berkeley_autolab_ur5 --fetch --revision c4e26a697fc4c04776b0558f83e563d14be0109f --max-download-mb 16 --json --output berkeley-preflight.json
```

This selected, already exposed development dataset has 28 episode/camera intervals that overrun their MP4 files. The detector reads MP4 headers, not decoded images. A next-file hint is a hypothesis: use the original clip and `recover-video-reference` to verify it before proposing metadata changes. The separate original-frame/native-loader qualification does not make header-only detection an identity proof.

Complete downloaded metadata and row files are checked against the listing's content IDs. Header byte ranges match the listed total size but are not checked against the full video's LFS digest. Offline records are not signed origin certificates.

`--max-download-mb` bounds metadata and Parquet body bytes. Header intake has a separate bounded request/byte/time budget reported in provenance. These are response-body accounting, not total network wire traffic. No weights, remote repository code or credentials are used.

## Local and CI use

```sh
roboclue advanced preflight ./dataset --policy ./checkpoint --deployment ./deployment.json --json
roboclue advanced preflight ./dataset --robot-command "$ROBOT_COMMAND" \
  --require deployment-units --fail-on error --json --output preflight.json
```

`--policy` is optional. An explicit deployment contract is compared when a checkpoint is supplied. A deployment contract without a checkpoint remains unassessed; it is not discarded. The contract is a declaration, not observed robot behavior.

Default exits are 0 for PASS/WARNING, 1 for ERROR, 3 for UNKNOWN and 2 for invalid invocation. `--fail-on error`, `warning` or `unknown` selects a CI threshold; warning includes errors and unknown includes all non-pass outcomes. The report retains the actual verdict. `--fail-on` counts every component status as well as every listed finding, so a component whose errors are capped or reduced still fails the gate. Automatically run components count toward these exits; `--no-groot-contract` skips the automatic GR00T dataset contract and `--no-units` the automatic units manifest. A report that would exceed 768 KiB first reduces its automatic components (keeping their status and counts); an explicitly requested check over the bound is refused (exit 2) with the remedy of a smaller `--max-files`.

A severity threshold alone does not require an unrequested comparison. For
example, generic `--fail-on error` can exit0 when missing evidence is UNKNOWN.
`--fail-on unknown` intentionally fails incomplete coverage, including healthy
data whose videos were not made available; it is not the default recommendation
for a scoped deployment-units gate.

Canonical `roboclue advanced preflight` validates an explicitly supplied robot
command/config and requires its units comparison to complete, unless units were
explicitly skipped with `--no-units`. Invalid robot options/types and missing
local dataset metadata are input errors. With `--require deployment-units`,
both launchers require the actual command/config, reject `--no-units`, and keep
missing required evidence at exit3 even with `--fail-on error`. A completed
required comparison can clear while unrelated video coverage remains UNKNOWN;
supported contradictions in the full report still fail. Generic legacy
`policy-doctor preflight` without `--require` keeps its historical severity-filter
semantics. Neither entry point's exit0 certifies whole-pipeline compatibility.

Rows and files are bounded by `--max-rows` and `--max-files`. The numeric scanner has its own maximum of 1,000 files, reported explicitly; episode checking supports up to 1,024. Limits do not imply complete arbitrary-size dataset coverage. Unsupported layouts and unavailable media remain UNKNOWN. `--no-video-headers` intentionally leaves video timing unassessed.

## MCP

Launch `roboclue advanced serve --root /absolute/path --allow-hub-datasets`, or print client configuration with `roboclue advanced setup --root /absolute/path --allow-hub-datasets`.

Use `preflight_check` with `source`, optional `checkpoint` and local `deployment`, optional immutable revisions, and `response_format`. The default concise response preserves component verdicts and next actions. Local paths are authorized before any request; misspelled argument names are errors. The existing JSON-only `--allow-hub` flag does not authorize Parquet/header access.

Concise responses are limited to 8 KiB. A version/recipe register preview shows the first ERROR or WARNING with its witnesses and names up to three other ERROR/WARNING findings by code, status and short message; a published-recipe difference keeps its "Results are not comparable to the published numbers." caveat. `other_actionable_omitted` counts any further ones and `findings_omitted` counts every finding not shown. Under size pressure, an evidence item can use `artifact_from` instead of repeating its file identity: the integer is the zero-based index of an earlier evidence item **in the same finding**. Read that item's path and file hashes together with the current item's own pointer and value. This shares the file identity, not the observation. Detailed responses retain direct file identities, but can fall back to a summary above 192 KiB; `truncated_from_bytes` identifies that fallback. Use CLI `--json --output FILE` when the full report is needed.

Preflight does not retain an unresolved dataset finding as a portable case yet. The existing incident and recovery tools support their documented retained criteria. Do not manufacture a repair to bypass this gap.
