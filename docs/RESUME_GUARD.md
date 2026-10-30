# Check before resuming a recording

Run the wrapper instead of `lerobot-record` when resuming an existing supported local LeRobot v3 recording:

```sh
python -m pip install '.[dataset]'
roboclue guard resume ./my-recording -- \
  lerobot-record --dataset.root=./my-recording --resume=true \
  --dataset.repo_id=YOUR_ACCOUNT/YOUR_DATASET \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

Supply the rest of your normal recording arguments as `--key=value`. The wrapper requires the console command `lerobot-record`, an explicit `--resume=true` and an explicit `--dataset.root` matching the checked directory. It rejects duplicate root/resume options, config-file overrides and inferred cache roots. Other native arguments are forwarded unchanged, without shell evaluation.

The wrapper reads the original directory before starting the recorder. It requires complete bounded rows/metadata, no detected resume hazard and the supported MP4 header checks. An ERROR, WARNING, UNKNOWN, missing evidence or exhausted budget blocks launch. A proposed repair plan for another copy cannot approve the original directory.

Inspect without starting any recorder:

```sh
roboclue guard resume ./my-recording --dry-run --json -- \
  lerobot-record --dataset.root=./my-recording --resume=true \
  --dataset.repo_id=YOUR_ACCOUNT/YOUR_DATASET
```

`--json` is restricted to dry runs so recorder output cannot be mistaken for the diagnostic JSON. A successful dry run says `launched: false`. A successful real run starts the explicitly supplied command and returns its exit status. Input errors exit2; supported faults/warnings exit1; incomplete required evidence exits3. A recorder terminated by a signal produces the conventional128+signal status.

## If the gate blocks

Follow the cited recovery evidence:

```sh
roboclue check ./my-recording --question recovery --video-headers --explain
```

Preserve the original. Apply reviewed recovery operations only to a separate candidate, including any required data-row rewrite. Rerun the same recovery check and your native loader on that candidate before using the wrapper with its path. The wrapper never performs a repair or deletes episodes. Increase `--max-rows` or `--max-files` explicitly if a complete supported scan exceeds the default budget; partial evidence is not a clean result.

## What the gate establishes

The checker models retained native LeRobot v3.0 recording behavior and cited source revisions, not every future framework version. Other layout labels are refused; relabeling a dataset is not a repair. It inspects numeric rows, episode ledgers, known resume hazards and supported video headers. It does not decode frames or establish calibration, policy quality, physical safety or task success. The downstream native recorder can still reject its own configuration or fail for reasons beyond these checks.

On POSIX systems the wrapper holds an advisory lock on the dataset directory through assessment and recording, without creating a file inside the dataset. Other RoboClue resume wrappers honor that lock. An unwrapped recorder or other process does not: stop other writers before using this gate. This is not an atomic transaction with LeRobot and cannot prevent a noncooperating process changing the data after assessment. Non-POSIX locking is not qualified.

This explicit wrapper is not automatically injected into an installed LeRobot command. For live training batches, use the existing [native guard attachment](GUARD_INIT.md); a static dataset check cannot replace the retained intended reference or first-batch target verification.

`roboclue advanced versions` reads this RoboClue Python environment's framework metadata without importing the framework. Version-only or different-commit installations are explicitly not revalidated; matching VCS metadata is not code attestation or a fresh runtime qualification. The recorder executable can belong to another environment, so the wrapper's JSON labels that context separately from the recording assessment. See [version context and maintenance](FRAMEWORK_VERSIONS.md).
