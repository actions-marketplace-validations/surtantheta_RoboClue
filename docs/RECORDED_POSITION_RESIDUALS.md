# Recorded-position residual triage

This opt-in preflight component compares two recorded columns of a LeRobot dataset: `action`, the recorded command, and `observation.state`, the recorded reading. It reports joints whose reading and command stay apart on frames where the arm is still. It looks for two patterns. In one, a joint's reading sits at one end of its travel while its recorded command keeps going past it. In the other, the two differ by a steady offset across the joint's travel.

```sh
roboclue advanced preflight ./so101_dataset --commanded-reach
```

A PASS means only that no residual pattern was found in the rows that were read.

## What it checks

- It runs only when `action` and `observation.state` hold the same motor positions: identical `<joint>.pos` names. Otherwise it reports UNKNOWN.
- A frame counts as still for a joint when its command and reading both change by less than 0.5% of the joint's range, or less than a small absolute floor, for 5 consecutive frames.
- **Held at an end (warning).** The reading sits at its highest or lowest values for at least about 6.7 seconds of still frames, while the recorded command is more than 5% of the joint's range beyond it.
- **Steady offset (warning).** Away from the ends of travel, the median residual is more than 5% of the range, over at least about 16.7 seconds of still frames.
- **Undecided (UNKNOWN).** These joints are not judged: mixed residuals with neither pattern, actions that do not follow the reading (deltas or velocities, for example), dead or idle joints, non-finite values, and too few still frames.
- Grippers and hand joints are measured but not judged, because grasping blocks them by design.

Rows are bounded by `--max-rows` (at most 200,000) and `--max-files`. A larger dataset is reported as UNKNOWN rather than partly judged. Reading Parquet rows needs the `dataset` extra (`pip install '.[dataset]'`); without it the component is UNKNOWN and names the extra.

## What it cannot tell you

LeRobot `146d847` records the teleoperator action before the robot's own processing and `max_relative_target` clipping (`src/lerobot/scripts/lerobot_record.py:325-327, 362-367`), so what the motors received is not recorded. A residual is triage evidence, not a calibration diagnosis. Each warning lists possible causes rather than asserting one:

- a range limit written during calibration;
- a software joint limit;
- a mechanical stop or obstruction;
- a stall under load;
- a leader arm with a wider range;
- slow tracking.

Neither a residual nor its absence establishes calibration, actuator delivery or robot health.

Known limits:

- A small-range joint (under about 15 degrees) held short of its command can pass.
- A stall on fewer than a quarter of still frames passes. The pass message states that share.
- Very slow tracking (a time constant of about one second) can read as a held end.
- Units are inferred from the readings: Feetech degree steps, a range of at least 10 units, or relative thresholds only. The basis used is reported for each joint.

Hub sources (`hf://`) give an explicit UNKNOWN for this component in this release. Download the dataset and pass its local directory.

## MCP

Pass `commanded_reach=true` to `preflight_check`. The concise response keeps the component's verdict, its residual statement, the row and episode counts, and the one-sentence limit on what recorded values can show. The full limits are on this page and in the detailed report's scope.
