# Set up a guard for your next training change

Start in the environment where your existing supported trainer already works:

```sh
roboclue guard init --framework groot --output ./my-guard
```

Choose `groot`, `lerobot` or `openpi`. This command creates a short setup guide,
a factory template and an attachment function. It imports no application code,
captures no reference and checks no candidate. Existing output directories are
refused so an old criterion cannot be replaced accidentally.

To capture your intended startup inputs, implement a zero-argument factory in
your own training project. It must construct the **independently reviewed
intended recipe before a change**, without calling training, the model or an
optimizer. Run this from that project, in its existing training environment:

```sh
roboclue guard init --framework groot \
  --factory training_setup:build_reference_context \
  --intent "Preserve the reviewed normalized relative-action startup recipe" \
  --reviewed-intended-recipe --output ./my-reference
```

The factory is explicitly executed local application code; RoboClue does not
sandbox it. The native capture helper reads startup inputs without a model
forward or optimizer step: one batch for GR00T/OpenPI, or the declared cohort for
LeRobot. It saves `startup-reference.json` and prints its criterion pin. It
consumes sampler state, so use a separate reference process.
The pin is also saved in `criterion-sha256.txt` for convenient transfer; **retain
it independently**, such as in your experiment record, before changing inputs.

Use the generated `attach_guard.py` function around the existing training or
policy call. Pass the independently retained pin. Keep the original criterion
and pin through diagnosis, correction and verification. Do not replace them
with a reference captured from the suspect candidate.

| Framework | Factory returns | Supported boundary |
| --- | --- | --- |
| GR00T | Configured native `Gr00tTrainer` before `train()` | Qualified N1.7 revision, one process, zero workers, `remove_unused_columns=False`, `eval_on_start=False`, `allow_padding=False`; [adapter guide](GR00T_TRAINING_GUARD.md) |
| LeRobot | Mapping with `batches` (already natively preprocessed deterministic startup batches) and retained content `dataset_id`; optional `padding`, `input_fields`, `declarations` | Preserve original `index` and native `action_is_pad`, or explicitly declare the supported padding objective. State/camera witnesses require declared input fields; [adapter guide](LEROBOT_TRAINING_GUARD.md) |
| OpenPI | Indexed native `DataLoaderImpl`, built from the final transformed dataset wrapped in `OpenpiSourceDataset` | Qualified synchronous PyTorch source, one process, zero workers; preserve the native model cast/device path and supply the actual model when attaching; [adapter guide](OPENPI_TRAINING_GUARD.md) |

A factory can return a context manager to release application-owned resources
after capture. Unsupported settings, missing source identity or failed capture
produce an error, never a candidate PASS. A partial output directory may remain
for diagnosis; choose a new one when retrying. Reference capture alone checks no
candidate. The adapter and core verifier retain their existing limits and exact
runtime/source bindings.

If a run already failed and no independent intent was retained, this setup
cannot reconstruct its historical target correctness. You can still inspect
current supported data contradictions and prepare a guard for the next change.
Later batches, policy quality and physical robot success remain unassessed.
