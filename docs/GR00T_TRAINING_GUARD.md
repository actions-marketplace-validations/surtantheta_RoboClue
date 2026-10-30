# Attach the startup guard to an existing GR00T trainer

The installed adapter observes original source samples before native mixture
shuffling and checks the native collator output and actual loss input. It does
not construct a replacement trainer, rewrite targets, or infer an intended
reference recipe. The supported source is GR00T N1.7 revision
`944fe70c4fd54e3fcb80a7d7258a328f9c061176`.

Use the product wheel in the existing training environment. Create a reference
explicitly from the intended recipe, in a separate process before the change:

```python
from roboclue import capture_gr00t_reference

# trainer is the already-configured native Gr00tTrainer.
contract = capture_gr00t_reference(
    trainer,
    output="startup-reference.json",
    intent={"objective": "Preserve the intended minmax-relative action targets"},
)
print(contract.criterion_sha256)  # retain independently before changing the pipeline
```

In the training process, surround the existing call with the adapter:

```python
from roboclue import Gr00tTrainingGuard

with Gr00tTrainingGuard(
    trainer,
    contract="startup-reference.json",
    criterion_sha256=retained_pin,
) as guard:
    trainer.train()

report = guard.summary()
```

No server binding or dataset-file hash manifest is supplied by the caller. The
observer discovers the dataset/processor from the existing native trainer,
fingerprints the visited source files and associates the original returned
sample objects with original episode rows and slices. Multi-sample shards and
mixture shuffling preserve those identities. The default observation cap is
16,384 prefetched samples; excessive shards/prefetch raise an actionable error
before training consumes the batch.

Supported settings are one process, `dataloader_num_workers=0`,
`remove_unused_columns=False`, `eval_on_start=False`, and native single-step
datasets with `allow_padding=False`. Attach before creating an iterator. Dataset
files must be regular materialized files; the current observer refuses input
symlinks and special files. The reference binds the verifier and NumPy version.
Preserve the declared runtime with the reference instead of recreating a bank
from suspected faulty targets.

`TargetContractViolation.report` identifies the source and first numeric/dtype
contradiction. The adapter adds observed processor-option differences, including
`use_relative_action`, as candidate explanations. Configuration differences alone
are not errors. Corrections must still pass the original numeric criterion.
UNKNOWN startup sources also block; later batches are explicitly unassessed.

Closing restores source and trainer hooks and releases retained sample/model
references. The reference helper owns and shuts down the iterator/prefetch it
creates, then publishes the bank. During ordinary training, the caller continues
to own native dataset/executor lifetime. Source signatures detect ordinary
concurrent changes; this is not historical provenance authentication.

## Native demonstration harness

The internal portable proof used a separately assembled wheel, hash-locked offline dependencies,
public pinned source, dataset derivative, model assets and scripts. That large packet is not bundled in this repository. It uses
Linux x86_64, CPython 3.12, Torch 2.9/CUDA12.6 and a suitable visible GPU. The
measured native profile uses FP32/math SDPA. Installation and execution are
separate checks; driver and shared-library requirements remain explicit.

```sh
python3.12 install.py --environment ../preview-env
../preview-env/bin/python -I -B verify.py --output ../native-trial
```

The verifier creates the intended native reference before any fault candidate,
retains its pin, and launches every arm in a fresh process. It requires native
faulty training to take two optimizer steps, guarded faulty targets to stop
before any forward/optimizer step, corrected targets and replay to train under
the original pin, and a healthy native control to match the recorded active
parameter slice. Unknown sources must block. Each native shard contains multiple
samples, exercising the reusable association rather than the original
one-sample-shard pilot. All inputs remain unchanged.

`train.py` shows the existing native trainer setup and optional adapter context.
It uses public Qwen pretrained frozen-backbone weights and a newly initialized
native GR00T head on recorded Arena simulation data. It is not an NVIDIA-trained
policy, a robot-success demonstration, or an independent human trial. The first
two-step proof is distinct from qualifying a whole training population or
measuring community adoption. The portable profile qualification (see [qualification summary](EVIDENCE.md))
records the completed offline installation, native runs, independent numerical
audit and eight successful gate terminals. Package qualification and native execution have distinct scopes. The demonstrated native reference/checker uses NumPy 1.26.4; run saved
capture checks in the matching runtime and keep the original pin unchanged.
