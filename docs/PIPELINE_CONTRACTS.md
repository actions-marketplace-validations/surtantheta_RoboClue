# Guard the initialization and camera inputs you actually use

These two checks address [openpi #804](https://github.com/Physical-Intelligence/openpi/issues/804)
and [LeRobot #5118](https://github.com/huggingface/lerobot/issues/5118).
They check separate, explicit contracts. A consistent result does not establish
whole-pipeline compatibility, successful weight loading, policy quality or robot safety.

## OpenPI: stop before a pretrained-load branch is silently skipped

In the supported PyTorch trainer, `weight_loader` does not select pretrained
loading. The resolved `pytorch_weight_path` controls that branch. Insert this
immediately before the existing trainer constructs its model:

```python
from roboclue import guard_openpi_initialization

guard_openpi_initialization(config)
```

`config` is the actual resolved `TrainConfig` already used by your application.
The helper statically discovers its source checkout for the supported native
config class. You can supply `runtime_source_root=OPENPI_SOURCE` explicitly.
It does not import the supplied source or read model weights.

With pretrained intent, a null `pytorch_weight_path` raises `PipelineContractError`
before training proceeds. Set that field to the converted PyTorch checkpoint
directory, which must contain `model.safetensors`, then repeat the same check.
The successful result says only that the load branch is selected: file contents,
tensor shapes and the actual subsequent loading result remain unchecked.

For intentional fresh initialization, use
`expected_initialization="scratch"`. A missing intent is not silently inferred
from a model name. Resume has a different restoration path and returns UNKNOWN;
this strict fresh-initialization guard does not approve a resume run. Unknown
source also raises with an UNKNOWN report instead of claiming a violation.

## LeRobot: stop a missing camera before it changes another camera's slot

For the supported pi0, pi05 and SmolVLA image preprocessors, present cameras are
packed first and blanks are appended afterward. A missing earlier camera can
move a wrist image into the front-camera slot. Install the guard on the existing
policy before calling its normal evaluation or deployment loop:

```python
from roboclue import install_lerobot_camera_guard

camera_guard = install_lerobot_camera_guard(policy)
# Continue using policy normally. The hook runs at its native image boundary,
# after the application's existing preprocessing and camera renaming.
```

The helper discovers the source of the policy's original bound preprocessing
method and requires an exact audited file profile. Installation checks that the
method belongs to this policy instance and that its active code matches the
method compiled from the audited source, without importing that source. A
different body, receiver, duplicate installation or later code replacement is
refused. Globals and the whole active process remain unattested. It reads config
order and actual batch key presence, without reading pixels. A leading/interior
gap raises before the native preprocessor consumes the images.

Restore the genuine missing camera under its trained key and repeat. Missing
trailing cameras keep their native padding/omission behavior; that slot check
can be consistent while camera adequacy remains unassessed. All cameras missing
is an error. For an intentional ablation, use a separately verified native
implementation that preserves slots. This guard never fabricates an unmasked
camera and does not patch, reorder or repair native image data.

`camera_guard.last_contract` contains the actual boundary's key declaration;
`camera_guard.last_report` explains the last check. `camera_guard.remove()`
restores the original method, provided nobody replaced the hook in the meantime.

## Use the same evidence in the CLI or MCP

Export resolved fields from the existing application; do not hand-copy a
different config or a batch before its camera renaming step:

```python
import json
from roboclue import capture_openpi_initialization

capture = capture_openpi_initialization(config, runtime_source_root=OPENPI_SOURCE)
with open("actual-initialization.json", "x") as stream:
    json.dump(capture, stream)
```

For cameras, save `camera_guard.last_contract` after its first call, including
after a caught `PipelineContractError`. Without a hook, call
`capture_lerobot_camera_inputs(config, actual_native_batch,
runtime_source_root=LEROBOT_SOURCE)` at the image-preprocessor boundary.
No pixels or model weights are stored.

```bash
roboclue check actual-initialization.json --fail-on error
roboclue check actual-cameras.json --fail-on error
```

The export may retain a local source directory. An explicit
`--runtime-source-root ACTUAL_SOURCE` overrides that directory when moving the
capture to another machine. Relative directories resolve beside the capture.
Every embedded or explicit source path requires authorization under MCP read
roots. The source is read as text; commands, model code and trainer code are
never executed by the CLI/MCP checks.

The existing default MCP call is:

```json
{"name":"preflight_check","arguments":{"source":"actual-cameras.json"}}
```

Or use `choose_workflow(goal="training", pipeline_contract="actual-initialization.json")`
to plan the call. The chooser performs no assessment. Detailed evidence is
available with `--json`, `--explain` or `response_format="detailed"`.

An ERROR returns exit 1 under `--fail-on error`; UNKNOWN does not become a PASS.
Use `--fail-on unknown` when a CI gate requires complete supported evidence.
The runtime Python guards fail closed for ERROR, WARNING and UNKNOWN.

## Supported source profiles and proof limits

- OpenPI PyTorch load selection: exact `scripts/train_pytorch.py` from
  `3e0325a5e6032a9ab6beef7395fe47ca724980a5`, load predicate at line 442.
- Camera-slot packing: exact pi0/pi05/SmolVLA model files from LeRobot
  `77710dee471d5cfb070fad1adc30c8f94dc14757` and the official 0.6.1 PyPI wheel.
  Checkout, site-packages and package-directory layouts are supported.

Matching a file hash identifies an audited source behavior, not every file in a
repository or the whole running process. A package version label alone does not
approve source behavior. New or modified source profiles return UNKNOWN.

Development proofs execute the unchanged native load block with a tiny real
PyTorch/safetensors model and unchanged native camera methods with controlled
calibration tensors. Public checkpoint metadata supplies the camera order.
Those proofs isolate load selection and camera packing; they do not construct
the full policies, run training, measure benchmark success or control a robot.

See [RC010](rules/RC010.md) and [RC011](rules/RC011.md) for source references.
