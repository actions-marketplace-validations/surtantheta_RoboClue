# Guard the batch LeRobot actually sends into the policy

`LeRobotTrainingGuard` attaches to the Torch module called by the existing
training loop. It checks after the original dataset, DataLoader, preprocessing
and device transfer, immediately before the policy's forward call. It preserves
the original batch, arguments, return value, gradients and optimizer behavior.
It does not replace native preprocessing or automatically rewrite a recipe.

The native development profile uses LeRobot revision
`146d8474932c8b566717eb2738ba459337ab71c4`. The final native profile executes
the original `update_policy` with genuine Accelerator 1.15.0 in a fresh CPU
environment. Its thirteen-phase matrix and independent numerical audit passed.
Installed-package checks passed separately; the full training CLI was not run.

## Declare and retain the intended startup cohort

Install RoboClue in the existing training environment. Torch and the robotics
framework remain that environment's dependencies; the adapter does not install
or launch them. Use a stable materialized dataset snapshot:

```python
from roboclue import fingerprint_training_dataset
from roboclue import capture_lerobot_reference

identity = fingerprint_training_dataset(dataset.root)
dataset_id = identity["dataset_id"]

# Use the existing loader and native preprocessor, with an explicitly selected
# deterministic startup cohort. The application owns its iterator/workers.
intended_batches = [preprocessor(raw_batch) for raw_batch in selected_batches]
declarations = {
    "units": {"action": None, "observation.state": None},
    "camera_map": {"observation.images.front": "declared_front_stream"},
}
criterion = capture_lerobot_reference(
    intended_batches,
    dataset_id=dataset_id,
    intent={"objective": "Preserve the reviewed startup recipe"},
    input_fields=("observation.state", "observation.images.front"),
    declarations=declarations,
)
criterion.save("startup-reference.json")
print(criterion.criterion_sha256)  # Retain independently BEFORE changing inputs.
```

Choose fields that exist in the actual native batch. Empty `input_fields` means
targets and padding only; camera/state preservation is then unassessed. Explicit
declarations can describe units, camera mapping or action meaning. Comparing
declarations detects a changed declared objective; it cannot establish whether
either declaration describes the physical robot correctly. `None` retains an
unknown physical unit rather than inventing radians from a coordinate name.

Exact state/image witnesses are appropriate for a deterministic pipeline and
the same covered source rows. They deliberately reject changed pixels, even
when a different augmentation might be reasonable. Do not freeze stochastic
augmentation as a deterministic objective. Either select an explicit repeatable
startup probe or exclude those fields and keep their coverage unassessed.

The bank has at most 128 distinct source rows, bounded numeric targets and a
4-MiB serialized contract. Input witnesses hash at most sixteen declared numeric
fields and 64 MiB per batch. A larger dataset does not enlarge this cohort's
coverage. The original global `index` column must survive preprocessing; sample
order, a batch counter and tensor shape are never substitutes for source IDs.

`fingerprint_training_dataset` hashes all regular files under the chosen root,
including metadata, videos and hidden files. Its default limits are 4,096 files
and 1 GiB; explicit bounds can increase these to 100,000 files and 16 GiB. It
refuses symlinks/special files, changed membership and changed files. A copied
snapshot has the same identity. A caller-supplied `dataset_id` remains a caller
declaration unless the application establishes it from actual contents. This
helper observes current bytes; it does not authenticate historical provenance.

## Attach at the actual policy boundary

In the training process, use the same retained snapshot, cohort selection,
runtime and reference pin:

```python
from roboclue import LeRobotTrainingGuard

with LeRobotTrainingGuard(
    policy,
    contract="startup-reference.json",
    criterion_sha256=retained_pin,
    dataset_id=dataset_id,
    declarations=declarations,
    batches=2,
) as guard:
    run_existing_training_loop()

report = guard.summary()
```

Attach to the actual outer module the trainer calls. Native `policy(batch)`
and `policy(batch=batch)` work; direct calls to `policy.forward` bypass Torch
hooks and are unsupported. Attach after other input pre-hooks. A later pre-hook
that could change checked inputs causes refusal. Application code remains
responsible for loader, model, process and optimizer lifetime.

ERROR and UNKNOWN both block before policy forward. Reports identify a source
row and first target/mask difference, changed input-field digest/schema or
contradictory declaration. A swapped camera can name the other retained field
whose digest matches; that is a source-linked observation, not camera calibration
or causal diagnosis. Correct the ordinary native recipe and retry with the
original bank and pin. Do not rebuild intent from suspected faulty output.

After the declared number of policy inputs, later calls continue and increment
`unassessed_forwards`. This is a bounded startup gate, not continuous full-run
certification. Uncovered rows within the checked scope block as UNKNOWN.

Native `action_is_pad` must be Boolean `[batch,horizon]`; true denotes padding.
The adapter broadcasts its inverse only as the comparison view and never changes
the native mask. `padding="all_valid"` is a separate explicit objective and
refuses a contradictory native padding mask. Keep the verifier's NumPy version
and exact adapter source with its criterion; runtime/source mismatches are
refused rather than silently reinterpreting an old reference.

## Workers and distributed execution

Worker processes do not host guard hooks. The parent policy hook reads global
row IDs carried in the returned batch. The public native development proof
includes two DataLoader workers and compares corrected named parameters with
the equivalent zero-worker run. This does not qualify arbitrary worker-side
transforms, mutable datasets or persistent-worker configurations.

For a supported DDP process group, initialize the group first and explicitly set
`distributed="collective"` on every rank. Every rank must use the same startup
scope and reach the same forward calls. A non-pass result on one rank causes
every rank to refuse before policy forward. The qualified development transport
is two-rank CPU/Gloo; NCCL/GPU, FSDP, uneven iterators, crashed ranks and process
recovery are separate unqualified profiles. Keep the trainer's normal process
timeouts and cleanup.

## Evidence boundaries

The initial public LIBERO profile uses the original reader, two decoded camera
streams, native ACT preprocessing and an 11,206,503-parameter randomly initialized
ACT. The original processor accepts a plausible wrong mean/std recipe; the bare
model takes two updates, while the guard blocks before forward. Corrected,
replayed, healthy and two-worker cases preserve the pre-fault criterion.

The final thirteen-phase matrix tests changed action/state normalization,
swapped camera fields, contradictory declarations, correction, replay, healthy,
two-worker and uncovered-source cases. Faulty action, camera and state inputs
each reach two updates without the guard; guarded counterparts reach zero
forwards and updates under the same pre-fault criterion. Corrected, replayed,
healthy and two-worker cases share their recorded named-parameter hash.

An independent NumPy/PyArrow audit rederived original Parquet windows, cooked
normalization, input witnesses and two AdamW updates of the 224-value action-head
weight parameter. Source normalization residual was zero; maximum independent
optimizer residual was 1.17e-10. The separate CPU/Gloo test uses a custom optimizer
loop with a rank-local batch repeated twice. These are not executions of the
complete training CLI. Matching named-parameter hashes exclude model buffers
and do not establish whole-model equivalence, learning quality, policy competence,
hardware success or community savings.
