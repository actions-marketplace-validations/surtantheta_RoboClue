# Run RoboClue where inputs change

Run explicit checks before the dataset, saved batch or configuration is consumed.
A diagnostic job passing means its declared checks and thresholds cleared; it
does not establish whole-pipeline compatibility or physical success.

## Use the GitHub Action

The repository's root `action.yml` is a reusable composite action for Linux
GitHub-hosted runners. It installs the checked-out action source, with dataset
dependencies, in its own Python 3.12 environment. It never installs a package
named in a dataset or executes a deployment command from the manifest.

After publication, use your reviewed RoboClue repository and immutable commit
in the `uses:` line below. `OWNER/ROBOCLUE@REVIEWED_COMMIT` is a placeholder,
not an already published action or release.

```yaml
name: Robot input checks
on: [push, pull_request]
permissions:
  contents: read
jobs:
  inputs:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - id: roboclue
        uses: OWNER/ROBOCLUE@REVIEWED_COMMIT
        with:
          manifest: .roboclue-ci.json
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: roboclue-evidence
          path: ${{ steps.roboclue.outputs.report-directory }}
```

This has been designed for downstream use; an isolated runner test is not
evidence that your exact public commit passed hosted GitHub Actions. Run that
gate after supplying the publication destination. Do not use `pull_request_target`
to run an untrusted contribution with privileged credentials.

## Declare the question, not an arbitrary command

Create `.roboclue-ci.json` in your consumer repository:

```json
{
  "schema_version": 1,
  "checks": [
    {
      "id": "deploy-units",
      "check": "deployment-units",
      "artifact": "recordings/pick-place",
      "lerobot_version": "0.6.2",
      "robot_command": "lerobot-record --robot.type=so101_follower --robot.use_degrees=false"
    },
    {
      "id": "training-config",
      "check": "config",
      "artifact": "configs/train.json",
      "reference": "reviewed/train.json"
    }
  ]
}
```

Each check runs through the public `roboclue check` implementation. The
`robot_command` string is parsed as data, never run. Paths are relative to the
manifest, stay inside the declared workspace and cannot contain `..`, symlinks
or special files. Unknown keys and criteria are input errors, not silently
ignored settings. A manifest contains at most 32 checks and is at most 256 KiB.

For `deployment-units`, optional `lerobot_version` declares the fleet's consumer
label. `0.6.2` selects the tested 146d847 SO100/SO101 source profile; it does not
attest the fleet's actual source bytes. Unqualified labels leave consumer units
UNKNOWN even with an explicit Boolean setting, so a required CI check cannot
silently inherit the pinned default. See [consumer profiles](CONSUMER_VERSION.md).
In that recorded profile, `lerobot-record` is collection-only and
`lerobot-rollout` is the policy-deployment command. A scoped units comparison
does not establish that the complete command can run.

| `check` | Required declaration | What clears the gate |
| --- | --- | --- |
| `deployment-units` | Exactly one `robot_command` or `robot_config` | Recorded units match the supported declared robot settings; unrelated video coverage is separate |
| `recovery` | Local supported LeRobot v3 recording | Required recording integrity assessment clears; proposed copies do not approve the unsafe original |
| `sim` | Source HDF5 `artifact` and candidate `converted` | The required source/export convention assessment clears; missing coverage remains blocked |
| `groot` | `groot_version` (`n1.5` or `n1.7`); optional `groot_config`/`groot_config_name` | The selected version/config contract clears, not decoded cameras or actual training |
| `conventions` | `openpi_config` and local reviewed `openpi_source_root` | Selected delta-transform consistency clears, not dataset repacking or loader-version compatibility |
| `config` | Candidate JSON and independently reviewed `reference` JSON | The saved values compare equal; semantic correctness and robot success remain unassessed |
| `training` | Saved capture, reviewed `reference`, and exactly one `pin_file` or `criterion_sha256` | Original target criterion verifies the capture; cannot reconstruct intent from a suspect batch |
| `dataset` | Dataset artifact; optional explicit public intake below | Bounded dataset assessment is complete without warnings/errors; this can block on missing video/statistics |

Retain reviewed references independently and protect their review in your
repository. A reference copied from the suspect candidate merely to clear a
check does not establish intended behavior.

Default `fail_on` is `error` for explicit consumer/config/training criteria;
required UNKNOWN still exits 3. You may choose `warning` or `unknown` for stricter
thresholds. Generic `dataset` checks require `fail_on: "unknown"` and cannot use
an error-only filter to approve incomplete evidence. Prefer the actual consumer
question when absent unrelated media is intentional.

`max_bytes`, `max_rows` and `max_files` can lower the defaults of 64 MiB,
200,000 rows and 1,024 files. Larger scans must be split into explicit bounded
jobs. The module also runs locally after installation:

```sh
python -m roboclue.automation run --manifest .roboclue-ci.json \
  --workspace . --output ./new-check-receipts
```

The output directory must be new and its parent must exist. It contains an
aggregate `result.json` and the original JSON evidence for each check. Checks
never modify the input artifacts. Exit 0 clears selected declared checks; 1 is a
supported finding/difference, 2 is invalid input, 3 is missing required evidence.
If several checks fail, invalid input takes precedence, then a supported error,
then UNKNOWN. A missing, unreadable or invalid artifact becomes that check's
structured exit2 result; the other structurally valid declared checks continue,
and every selected individual result is retained. An invalid manifest schema
starts no checks. When its explicitly requested output directory is new and
safe, it still receives `result.json` explaining the global input error. Invalid,
existing, symlinked or input-contained output paths are never overwritten merely
to write an error receipt.

## Public Hub data requires two explicit permissions

Use a `dataset` check with `artifact: "hf://ORG/DATASET"`, `fetch: true` and an
immutable 40-character lowercase `revision`. Also pass `allow-hub: 'true'` to
the action, or `--allow-hub` to the local runner. Both are required. This permits
bounded public artifact intake, not full-video/model downloads. Python package
installation by the action separately requires normal package-index access.
No consumer compatibility is inferred from a Hub dataset name.

## Pre-commit: small reviewed JSON files

Pre-commit is for a fast config/metadata comparison. It does not scan a large
dataset or intercept training. Put explicit candidate/reference JSON pairs in
`.roboclue-precommit.json`:

```json
{
  "schema_version": 1,
  "checks": [
    {
      "id": "camera-config",
      "check": "config",
      "artifact": "configs/cameras.json",
      "reference": "reviewed/cameras.json"
    }
  ]
}
```

The hook refuses other check kinds and files larger than 1 MiB. Changed
configured candidates, references or the manifest select the comparison;
unrelated JSON is explicitly unselected, not certified. Standard pre-commit
checks the working-tree files it supplies, not an independent historical
snapshot of the Git index.

After publication, configure your reviewed repository/commit:

```yaml
repos:
  - repo: https://github.com/OWNER/ROBOCLUE
    rev: REVIEWED_COMMIT
    hooks:
      - id: roboclue-config
```

Install the local Git hook and run its first comparison explicitly:

```sh
python -m pip install pre-commit
pre-commit install
pre-commit run roboclue-config --all-files
```

The configuration file alone does not install a Git hook. Keep the reviewed
reference files and manifest in the repository before enabling it.

For a local checkout you can use a `repo: local` hook with `language: system` and
`entry: python -I -m roboclue.automation pre-commit --manifest .roboclue-precommit.json --`
after installing RoboClue. To choose another manifest, replace that entire entry;
the terminal `--` prevents filenames from becoming command-line options. The
isolated interpreter prevents a consumer-local `roboclue.py` from shadowing the
installed diagnostic package.

These are downstream CI/pre-commit gates. Existing native first-batch guards
and opt-in recording/startup gates address later risk boundaries; these checks
do not silently monkey-patch installed robot frameworks.

## Weekly upstream review

`.github/workflows/upstream-revalidation.yml` runs weekly and on manual request.
It replays portable adapter/diagnostic tests and observes the current upstream
HEAD and latest release label for the four cataloged frameworks. A changed or
unavailable source identity blocks the job and retains an actionable artifact.
It does not open issues, publish releases or silently extend compatibility.

A matching source pin only establishes an observed identity. Portable tests,
installed distribution versions and source observations do not replace native
framework qualification. Before extending support, review the affected rules,
renew their source/adapter/native execution evidence and update the catalog's
explicit tested identities. The weekly job does not run GPU trainers or certify
the latest release from its version number.
