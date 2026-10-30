# Saved native capture replay

Run the target/mask guard on genuine retained GR00T captures using the core
NumPy installation. No model weights, GPU, native framework, Arrow, MCP server,
network access or old filesystem paths are needed at runtime.

```bash
python -m pip install .
roboclue advanced demo-native --output ./roboclue-demo
```

The output directory must be new and its parent must already exist. Existing
files, directories, FIFOs, symlinks and paths through symlink parents are refused.
The command does not overwrite a previous run. If a run fails after creating its
output, keep that evidence and choose a different new output directory.

For module use, `python -m roboclue advanced demo-native --output ./roboclue-demo`
performs the same replay through the canonical entry point.

## Read the result

The command prints a short result with all four control verdicts, the fault
counts, the retained pin and paths to the saved reports. Exit 0 means the expected
replay controls reproduced; the wider research question remains UNKNOWN.
The saved `summary.json` has `status: completed` and `overall_status: unknown`.
For the complete summary and provenance on stdout, add `--json` and use another
new output directory:

```bash
roboclue advanced demo-native --output ./roboclue-demo-json --json
```

Both output modes save the same full summary and actual guard reports:

| Saved capture | Actual guard result | Meaning |
| --- | --- | --- |
| Healthy | `pass` | Both covered samples preserve the declared target/mask reference. |
| Fault | `error` | 1,400 target values differ; both samples also have a mask dtype mismatch (`float64` expected, `float32` observed). |
| Corrected | `pass` | The retained corrected capture preserves the same reference and pin. |
| Uncovered | `unknown` | Both original source identities fall outside the reference bank; pairing is not guessed. |

A selected target-preservation PASS covers only these identified saved samples.
The wider research question remains UNKNOWN: the replay establishes no model
learning, new native execution, physical correctness, historical intended recipe
or correctness of another dataset population. It makes zero model forwards and
zero optimizer steps. The earlier native qualification's two optimizer steps
are historical provenance, not work performed by this command.

Open `summary.json` for the result, source provenance and artifact paths. The
four `*-report.json` files are actual `TargetContract.check` reports. They retain
the per-source findings, first differing coordinate and independent pin.
`reference-contract.json` and `criterion.sha256` are written before any candidate
check. `inputs/` contains the authenticated public target-only inputs, manifest,
source provenance, attribution notice and dataset license.

## Source identity and transformation

These fixtures come from the qualification retained on 2026-10-26, job
`136277.caquelon`, and the saved-source audit, job `136258.caquelon`. The audit
checked native capture array hashes against source derivation; this replay
verifies the packaged bytes and exact array identities, and compares the saved
targets. It does not repeat that source-data audit.

The source is NVIDIA Corporation's
[Arena-G1-Static-PickNPlace-Task](https://huggingface.co/datasets/nvidia/Arena-G1-Static-PickNPlace-Task/tree/37ba80a99486a4c308477854f54b4e82e77777dc),
revision `37ba80a99486a4c308477854f54b4e82e77777dc`, recorded simulation data.
It was converted by IsaacLab-Arena converter
`8b4a3a47fc53de23e8205089d71109a2e2348acd` and processed by native GR00T
`944fe70c4fd54e3fcb80a7d7258a328f9c061176`.
The [pinned dataset card](https://huggingface.co/datasets/nvidia/Arena-G1-Static-PickNPlace-Task/blob/37ba80a99486a4c308477854f54b4e82e77777dc/README.md)
and [upstream license](https://huggingface.co/datasets/nvidia/Arena-G1-Static-PickNPlace-Task/blob/37ba80a99486a4c308477854f54b4e82e77777dc/LICENCE)
identify CC-BY-4.0. The exact upstream license and an attribution/change notice
are bundled alongside these data. RoboClue software retains its separate
repository software license. No NVIDIA endorsement is implied.

The original native criterion is
`758d5c621e178a3d5cfbda6995a02b1fbd35b00cf1e695d41815929e52b5f898`.
It requires its original verifier source and NumPy 1.26.4. Those requirements
remain strict. The demo creates a **different criterion for the installed
verifier and NumPy runtime**, using only the frozen declared reference. It
never treats a fault or correction as a new intended reference.

`reference.json` is a declared target-only projection of that original bank:
it preserves the intent, source keys, exact action/mask descriptors and complete
per-source provenance, and omits the original criterion's schema/kind/verifier
envelope. The projected reference hash is
`5091814bfc10ed09038d2e69d25bd89254f4a5ac0a8612c77bca4439b303b790`.
The healthy, fault, corrected and uncovered files preserve the exact retained
raw capture bytes, source keys, array values and native dtypes. Healthy and
corrected are equal bytes retained from separate original phases; the native
fault and guarded fault also have equal original captures. The public relative
data/video paths in provenance identify original associations; the actual
dataset, images, model parameters and native runtime are not bundled.

`inputs/provenance.json` records original artifact identifiers and raw hashes,
qualification/audit receipt hashes, source versions, license identity and each
transformation. `inputs/manifest.json` records the projected file hashes. The
module pins that manifest and verifies every fixture before creating a criterion.
The original inputs and old criterion are never modified. There is no relaxed
comparison tolerance, ignored schema mismatch or reference rewrite after fault.

## Use your own intended recipe

This replay is a short demonstration of an already captured example. For your
run, declare the intended recipe and capture its native post-collator action
and mask targets before changing preprocessing; retain its criterion pin
independently. See [the GR00T training guard](GR00T_TRAINING_GUARD.md)
for the prospective native capture boundary. Missing intended-recipe history
cannot be reconstructed by freezing the current suspect batch.

Existing generic saved-target tools can read a demo output without native
execution. Use the paths returned by the command: `reference_contract`,
`criterion_pin`, and the healthy/fault/corrected/unknown JSON files in `inputs/`.
The demo's reference projection is deliberately a separate input format; use
`reference-contract.json` as the saved criterion, not `inputs/reference.json`.
