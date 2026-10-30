# Units, frames and timebase

LeRobot writes each vector feature's dtype, shape and coordinate names to `meta/info.json`, but never its unit or the motor normalization it was recorded in. The robot decides the unit at run time. SO, Koch, OMX and LeKiwi arms use degrees when `use_degrees` is set and -100..100 across the calibrated range otherwise, so the same numbers can mean different angles on another robot setup. Preflight infers the recorded mode from the values and compares it with the robot that will consume them.

```sh
roboclue check ./so101-dataset --robot-command 'lerobot-record --robot.type=so101_follower ...'
roboclue check ./so101-dataset --robot-config rollout.json
roboclue check ./so101-dataset --units
```

The `units` component runs with `--robot-command` or `--robot-config`. Use
`--units` for the dataset-only manifest; it does not establish a deployment's
units without the intended robot configuration. The detailed
`roboclue advanced preflight` interface also supports `--no-units`.

For an explicit CI requirement:

```sh
roboclue check ./so101-dataset \
  --robot-command 'lerobot-record --robot.type=so101_follower ...' \
  --require deployment-units --fail-on error --json
```

Missing or unknown required evidence fails this gate. It cannot silently approve
deployment merely because no contradiction was found.

## What it checks

- **Joint mode.** The recorded values of each arm are tested against the grids that LeRobot `146d847`'s motor bus produces: degrees, -100..100, or the pre-PR #777 calibration. A mode is stated only when enough distinct values sit on exactly one grid.
- **Deployment.** The robot type and `use_degrees` are read from the command or config text. A `use_degrees` left unset takes the default of LeRobot `146d847`; other LeRobot versions may default differently. A recorded mode that differs from the robot's mode is an error.
- **Quaternions and timebase.** Unit norm is checked for named quaternions, and the median timestamp step is compared with `1/fps`.

## What it cannot tell you

- The policy (`--policy.path`) and its processors are not read. Remedies assume the policy was trained on this dataset's values as recorded.
- Calibration files and the arm are not inspected, so calibration offsets between arms are not checked.
- Radians, metres and other non-motor units are never asserted. Values that fit two grids stay unknown.
- Reading the rows needs the `dataset` extra. Without it, the component reports that the values were not read and names the extra.
- The component is bounded (128 KiB): every finding is kept, with its text shortened when very long; motor joints keep their per-coordinate records, while wide non-motor features (environment state, embeddings) list their first coordinates and count the rest by status and unit (`coordinates_summary`). Run automatically, it never makes a preflight refuse at the 768 KiB report bound: it is reduced, then cut to a short stub that keeps its status and counts, and only as a last resort left out with its status and the reason in `coverage`.
- Hub sources (`hf://`): `--units` gives an explicit UNKNOWN, and with a robot selector the report says the manifest was not run. Download the dataset and pass its local directory.

## MCP

`preflight_check` takes `units`: `true` runs it, `false` skips it, and the default `null` runs it with `robot_command` or `robot_config`. The concise response keeps the component's verdict, what it was compared with, and these limits.
