# OpenPI startup targets at the loader and actual Torch model input

`OpenpiTrainingGuard` checks the first declared native PyTorch batches and,
optionally, the actual `actions` supplied to a Torch model before its forward
method executes. Source identities come from original random-access dataset
indices and a portable content fingerprint, never batch ordinals. The adapter
supports the original OpenPI revision
[`215abfb`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)
on the qualified synchronous PyTorch path.

This is target-preservation evidence under declared intent. It does not infer
that a normalization profile, action convention, dataset slice or reference
target is appropriate for a trained policy or physical robot.

## Prepare an intended reference before the suspected change

Start with the application's existing **final native `TransformedDataset`**.
Wrap it before constructing its original `TorchDataLoader` and `DataLoaderImpl`:

```python
from openpi.training.data_loader import DataLoaderImpl, TorchDataLoader
from roboclue import (
    OpenpiSourceDataset, capture_openpi_reference,
)

identified_dataset = OpenpiSourceDataset(
    transformed_dataset,
    dataset_root="/path/to/stable-materialized-dataset",
)
native_loader = DataLoaderImpl(
    original_data_config,
    TorchDataLoader(
        identified_dataset, local_batch_size=2, framework="pytorch",
        num_workers=0, shuffle=True, seed=42,
    ),
)
contract = capture_openpi_reference(
    native_loader,
    output="new-reference-contract.json",
    intent={"objective": "preserve my intended normalized delta-action startup targets"},
    batches=2,
)
print(contract.criterion_sha256)  # Retain independently before changing anything.
```

Use a separate reference process. Reference capture consumes sampler state and
owns the iterator it creates; it restores the original loader before publishing
the new criterion file. It invokes no policy model or optimizer. A new criterion
must not be generated from a suspected faulty candidate simply to clear it.

Content identity covers all regular files below the declared materialized root,
including metadata and caches. The default bounds are 4,096 files and 1 GiB;
explicit bounds may be increased to 100,000 files and 16 GiB. Symlinks and special
files are refused. A relocated identical snapshot retains its identity, while
changed files or membership invalidate it. The index denotes this dataset's
original random-access index; it is not automatically a physical Parquet row,
an authenticated source history or an arbitrary subset's original global index.
Keep the dataset's index mapping stable and declare any custom selection recipe.

## Check the changed pipeline and actual model input

Construct the changed native pipeline using the same source envelope and original
sampler recipe, then attach to its existing Torch model:

```python
import jax
import torch

from roboclue import OpenpiTrainingGuard

with OpenpiTrainingGuard(
    changed_native_loader,
    contract="reference-contract.json",
    criterion_sha256=independently_retained_pin,
    batches=2,
    model=existing_torch_model,
) as guarded_loader:
    data_iter = iter(guarded_loader)
    observation, actions = next(data_iter)
    # Keep the application's existing, pinned native device/cast steps:
    observation = jax.tree.map(lambda value: value.to(device), observation)
    actions = actions.to(torch.float32).to(device)
    losses = existing_torch_model(observation, actions)

report = guarded_loader.summary()
```

The adapter owns its created iterator and closes it on context exit, including
the common held `data_iter` and early-break cases. It restores the original
native loader, removes its model hook and releases those references. Cleanup
errors preserve an already active training exception and add a failure-type note.
`data_config()` forwards the native configuration while the guard is open.

The exact native transforms, collator and `Observation.from_dict` execute.
Reserved numeric source metadata is attached **after** native repacking and
transforms, checked for collisions, carried through native collation, and removed
before `Observation.from_dict`. Action/state/image tensors are not rewritten to
make a check pass. The original native `(Observation, actions)` tuple is delivered.

The model pre-hook checks actual post-cast/device targets. It uses source keys
retained from the checked native delivery, rather than deriving them from a model
call counter. Follow the native **sequential** delivery/consumption order.
Unpaired model calls remain UNKNOWN and block. Reordered, swapped or duplicated
batches with different target values fail the retained check. This is not origin
authentication for arbitrary equal-valued tensors or asynchronous consumers.
Attach other model pre-hooks before RoboClue; a later pre-hook is refused during
the checked scope so it cannot alter targets after RoboClue checks them.

Without `model=`, the guarantee ends at loader delivery: subsequent caller
changes are not checked. With the hook, only the declared action inputs are
checked; state, image meaning, cameras, arbitrary model internals and output
behavior remain outside this target criterion. Later batches/model calls are
explicitly counted as unassessed and continue normally after the startup scope.

## Exact raw and model criteria; no invented native loss mask

The pinned original
[`train_pytorch.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train_pytorch.py)
casts actions to float32 before moving them to the model device. Its SHA256 is
`baa66f1642e955d5cc2ec3f54d9495035df61d71b028b98ab4ebb1e240548311`.
The criterion binds this prospective projection and this exact adapter's hash.
It stores two entries per source sample: the exact native-loader value/dtype and
the declared float32 model value. The actual model input is **not** recast by the
checker to make it pass. Wrong input dtype or a changed value fails.

OpenPI supplies actions without the GR00T-style `action_mask`. RoboClue supplies
an explicitly declared **all-true comparison witness** only to select every
action value for the existing `TargetContract`. This is **not a native OpenPI
action or loss mask**. It is never injected into the native model, used to alter
the loss, or presented as evidence of native masking semantics. The original
TargetContract verifier and its exact NumPy/core identity remain unchanged.
With two entries per source, its existing 128-record limit permits at most 64
distinct source samples, also subject to its 4 MiB criterion limit.

## Coverage and qualification boundary

Supported startup configuration is PyTorch, one process, zero DataLoader workers,
the original native classes/collator and the pinned source checkout including
its training script. JAX/sharded arrays, worker processes, DDP, RLDS and other
revisions are explicitly unqualified and refused. There is no MCP training or
robot-control launch operation.

The public CPU example is `examples/openpi_training_guard/public_cpu.py`. It
reads a user-supplied local LeRobot-format Arena simulation export and a pinned
OpenPI checkout; it has no server binding. Use its `reference` phase first, retain
the returned pin, then pass that contract/pin to candidate phases. It requires
the native dependencies of an already working OpenPI PyTorch setup, plus PyArrow
and PyAV for its public Parquet/video fixture.

The fixture intentionally selects the first 32 of 43 recorded coordinates and an
eight-step horizon, derives normalization statistics from actual public values,
and runs the original Repack/DeltaActions/Normalize/ResizeImages/TorchDataLoader/
Observation path. This is an explicitly declared arithmetic profile, not a claim
that this selection or normalization is suitable for a trained Pi0 policy.
The model-input probe is a **controlled real `torch.nn.Module` consumer**; it is
not Pi0, a trained policy, a simulator or an optimizer run. Native Pi0 forward,
optimizer behavior and physical benefit require separate qualification.

The retained CPU matrix uses two checked batches of two samples, followed by
one explicitly unassessed batch. Its raw loader targets are float64 and the
prospectively declared model targets are float32. Under criterion
`4f00bc4c578ca478e01e5077d1efef895721d1cb2f7f46a6b0d03de1c673c5c4`,
the original native pipeline accepts three wrong-target batches; the guard
detects 512 differing values before native Observation delivery. A separate
one-value mutation after delivery is blocked before the consumer's first forward.
Wrong model dtype, reordered/swapped/duplicate consumption, an unpaired model
call and a later pre-hook also exercise refusal. Correction, healthy control and
fresh-process replay have identical recorded action/state/image witnesses.

A separately declared legal absolute-action objective passes under its own
criterion; it never replaces the original delta-action criterion to clear the
fault. A separate NumPy/PyArrow audit rederives all 2,048 raw/model reference
values exactly from the selected public source and verifies the float32 projection
without importing RoboClue, OpenPI or Torch. Base unit environments pass 32
combined adapter/core cases; six optional real-Torch unit cases are explicitly
skipped where Torch is absent and covered by the corresponding real-Torch CPU
matrix controls. These measurements establish the stated software boundaries,
not Pi0 training, model quality or independent researcher adoption.
