# What do I already have?

Choose by the evidence you have, before building a manifest or attaching a guard.
RoboClue can inspect supported files now; comparing a change against intent needs
an independently reviewed reference retained before that change.

Ask for a plan without running a check:

```sh
roboclue advanced start --goal training --framework groot
roboclue advanced start --goal dataset --dataset /absolute/path/to/dataset
```

The chooser reports which inputs are present, what is missing and the next
commands. It does not open a trainer, fetch a dataset, validate file semantics or
run the suggested assessment. `inputs_present` means the required paths/pin were
supplied, not that the evidence is valid. A produced plan exits 0 even when you
still need evidence; invalid input exits 2.

| I already have… | Useful now | What I still need |
| --- | --- | --- |
| A local dataset, metadata or configuration | Check supported file/row contradictions and declared configuration with a health card or preflight. | A clean result covers the named checks, not correct action meaning or training targets. |
| A reviewed reference, its independently retained pin and a candidate native capture | Compare covered targets/masks, cite the first difference, then verify a correction under the same criterion. | Source identities, supported framework settings and the reference's verifier/NumPy runtime must match. |
| A suspicious training run but no reviewed reference | Inspect supported artifacts and plan a prospective reference capture. | Someone must justify the intended convention. A baseline generated from the suspect candidate cannot answer whether that candidate was correct. |
| A runtime recording with declared units, clock and association evidence | Triage supported software-boundary events. | Motor delivery, contact/force truth and physical success need separate evidence. |
| Owned files and an old/new framework version | Inventory files and observe a public upstream revision; compare declared configurations. | Asset impact requires a supported native comparison against retained intent. A version change alone remains advisory/UNKNOWN. |
| A source recording and a converted export | Check supported source/export associations against an explicit recipe. | Arbitrary conversions or missing original intent cannot be reconstructed from filenames. |

## Check files you have now

For a local supported dataset, install the dataset extra in your demo environment:

```sh
python -m pip install '.[dataset]'
roboclue advanced health-card /absolute/path/to/dataset --output ./my-dataset-health.json --json
```

This generic card omits the GR00T dataset contract. `--consumer` is a freeform
label, not a compatibility declaration. Select that contract with
`--groot-contract` (MCP `groot_contract=true`) or explicit GR00T recipe/version
configuration. A generic card cannot establish native training compatibility.
Use `roboclue check DATASET --question groot` for the GR00T dataset/configuration
question, and select the intended recipe/version explicitly. The full
`roboclue advanced preflight` interface retains its detailed existing options.

The report names assessed files/rows, findings and coverage. A cited contradiction
is a concrete investigation lead. A healthy component means its check passed;
UNKNOWN means another question lacks evidence or coverage. Neither status proves
that the intended action convention was right. For detailed processing checks,
see [preflight](PREFLIGHT.md) and [health-card examples](HEALTH_CARDS.md).

## Preserve intended training inputs for the next change

Review a small source-row example and your intended action convention with the
pipeline owner. Capture the supported native boundary from that reviewed recipe,
retain its criterion pin independently, and keep that reference while testing the
candidate and correction. Start with your integration guide:
[GR00T](GR00T_TRAINING_GUARD.md), [LeRobot](LEROBOT_TRAINING_GUARD.md) or
[OpenPI](OPENPI_TRAINING_GUARD.md).

An offline comparison needs all three artifacts:

```sh
roboclue advanced check-training-targets --contract startup-reference.json \
  --criterion-sha256 INDEPENDENTLY_RETAINED_SHA256 --capture native-capture.json --json
```

Do not replace the original criterion to make a failing candidate pass. A newly
reviewed reference helps future changes; it does not recover missing historical
intent. The [small CPU example](QUICKSTART.md) illustrates the comparison boundary
before you integrate a real trainer.

## Retain evidence during ordinary work

- Runtime: attach the [recorder](RUNTIME_RECORDER.md) to the software calls you already make; retain declared units, timebase and command/feedback association.
- Simulation/evaluation upgrades: retain commits, configuration, source bytes and reviewed native before/after captures. Use the [upgrade watch](UPGRADE_WATCH.md) only after checking its catalog and evidence requirements.
- Conversion: keep original source, recipe and reviewed interpretation alongside the export. See [conversion criteria](CONVERSION_REFERENCE.md).

These steps preserve evidence for future checks. They do not settle a past
training run, evaluation or physical event whose required evidence was never
recorded. [Result meanings and supported profiles](SCOPE.md).
