# Action convention

LeRobot's relative-action step, openpi's DeltaActions and GR00T's RELATIVE representation all compute `action[i] - state[i]` for each masked coordinate. That is meaningful only when `action[i]` is an absolute target of the same quantity, in the same units, as `state[i]`. Preflight fingerprints the recorded numbers (absolute, incremental or undecided, per coordinate) and compares them with the consumer that will convert them.

```sh
roboclue check ./libero-dataset --openpi-config pi0_libero --openpi-source-root ./openpi
roboclue advanced preflight ./dataset --policy ./lerobot-checkpoint      # a checkpoint with a relative-action processor
roboclue advanced preflight ./dataset --policy ./groot-checkpoint --groot-embodiment libero_panda
```

The component runs when the inputs name a consumer it models:

- a local LeRobot checkpoint whose saved processors convert actions to relative ones. A checkpoint without such a step does not trigger it, so an ordinary `--policy` preflight is unchanged;
- a local GR00T checkpoint folder with `processor_config.json`. Pass `--groot-embodiment` when the config declares several tags. A `processor_config.json` that links out of the checkpoint folder is not read, and the component reports UNKNOWN;
- an openpi builtin TrainConfig (`--openpi-config NAME`) read as text from a pinned openpi `215abfb` checkout (`--openpi-source-root PATH`). Nothing is imported or run.

## Reading the result

The display keeps two questions apart: `delta transform consistency: pass;
dataset fingerprint: partly undecided` means the declared delta conversion fits
the coordinates the data decides, while other coordinates remain undecidable
from the numbers. It does not establish matching observation keys, framework
versions or native training compatibility. The component status, and with it the
overall status when no independent error is found, stays UNKNOWN. LIBERO under `pi0_libero` is the common example:
the extra DeltaActions on LIBERO's delta actions is documented by openpi, while
roll, pitch and yaw coordinates stay undecided.

An ERROR needs a declaration that the data agrees with, such as coordinates named as velocities that a relative step converts as positions. An inferred convention plus a transform gives a warning.

## OpenPI input keys before delta conversion

Selecting `--openpi-config` and `--openpi-source-root` also runs the separate
`openpi_context` component. It reads the selected TrainConfig and the raw input
names in its literal RepackTransform from fingerprint-verified source. It handles
the Libero, Aloha and LeRobot DROID data factories, including Aloha configs that
override their default camera mapping. It does not execute that checkout.

For example, `pi0_libero` requires `image`, `wrist_image`, `state` and `actions`.
Its declared `prompt_from_task=True` supplies a separate prompt declaration;
the static check does not require a raw `prompt` feature or verify that the task
lookup succeeds. A dataset declaring `observation.images.image`,
`observation.state` and `action` receives a named raw-key ERROR even if its delta
transform choice is consistent. Use a matching dataset/config or explicitly
adapt the selected repack mapping. Renaming a public dataset blindly is not a
repair.

The component reports the dataset's `codebase_version` beside the LeRobot git
dependency pinned in the verified OpenPI `uv.lock`. These are different facts:
the lockfile is not evidence of an installed loader, and the static check does
not claim loader-version compatibility or incompatibility. The original action
fingerprint and delta-transform verdict remain separate and unchanged.

RLDS DROID recipes are identified as a different backend; LeRobot metadata
cannot establish their raw inputs. Configs supplied dynamically outside the
verified manifest, data factories without a literal repack contract, and source
fingerprint drift remain UNKNOWN. A raw-key PASS means the declared required
names are present. It does not verify decoded images, native loading, later
transforms, normalization or model training.

## What it cannot tell you

- Only recorded numbers and names are used. No model, consumer code or robot is run.
- Delta and velocity differ only by a scale; base vs end-effector frames and Euler vs axis-angle are not decidable from values.
- `absolute` allows a ±15% scale tolerance, so it does not prove identical units.
- Hub datasets and Hub checkpoints give an explicit UNKNOWN in this release.

## MCP

`preflight_check` takes `openpi_config`, `openpi_source_root` (inside the read roots) and `groot_embodiment`. The concise response keeps the consumer and dataset-fingerprint verdicts, the separate OpenPI raw-key component status and its findings, and these limits. Request `response_format="detailed"` for the complete key mapping, source witnesses and dependency pin.
