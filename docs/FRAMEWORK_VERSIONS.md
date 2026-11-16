# Installed framework context

```sh
roboclue advanced versions
roboclue advanced versions --json
```

RoboClue observes the metadata of `lerobot`, `isaaclab`, `gr00t` and `openpi`
distributions in **its own Python process**. It does not import the frameworks,
run their code, inspect arbitrary dataset-provided paths or fetch releases.
A recorder/trainer launched by another executable can use a different environment.

| Source identity | What the metadata establishes |
| --- | --- |
| `declared_tested_commit` | Its full Git commit equals a catalog-tested pin. Installed code bytes and current execution were not authenticated or rerun. |
| `not_revalidated` | A version or other source commit was observed; the catalog has not established this installation's behavior. |
| `not_installed` | No matching distribution was discovered in RoboClue's Python environment; another consumer environment can still contain it. |
| `unknown` | Metadata is absent/malformed, ambiguous, linked, unsupported or outside the read bounds. |

None of these labels means pipeline compatibility PASS. Semantic versions can
identify several different source states, and editable metadata can describe a
checkout without attesting its current commit. Exact commit metadata is still a
declaration, not a signature or proof that installed files are unchanged.

The distribution names are sourced from the pinned project metadata:
[LeRobot](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/pyproject.toml),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab/pyproject.toml),
[GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/pyproject.toml),
[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T/blob/944fe70c4fd54e3fcb80a7d7258a328f9c061176/pyproject.toml)
and [openpi](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/pyproject.toml).
The client-only `openpi-client` distribution does not identify the training
package. The GR00T package uses two zeroes: `gr00t`, not `groot`.

Python integrations can import `collect_framework_versions` or
`supported_framework_profiles` from `roboclue`.
The latter supplies repositories and exact tested commits for deliberate upstream
revalidation workflows. Neither function widens a detector's supported versions
or changes its numerical verdict.

## Weekly upstream review

The `upstream compatibility review` GitHub workflow runs two independent jobs:

- **Portable adapter and diagnostic tests** fail when a regression test fails.
- **Upstream source observations (review alerts)** compare each repository's
  current HEAD with the recorded qualification commits. Known source changes
  produce warnings and a job summary with comparison links. Missing or invalid
  source identities fail the monitor, so unavailable evidence cannot pass silently.

A green source-monitor job means the observations completed, including any
pending review warnings. It does not establish compatibility with newer code.
The JSON artifact retains `revalidation_required` when a source changes, and
native qualification remains `not_rerun`. Review the affected code and rerun its
source, adapter and native qualification before extending support or changing
the recorded pins.

For a strict local check, run:

```sh
python scripts/revalidate_upstream.py --fetch --output upstream-observations.json
```

The default command still returns 3 for changed or unavailable source identities,
2 for input/output errors, and 0 when all identities match recorded pins. The
workflow adds `--github-actions` to report known changes as review warnings;
unavailable source identities return 2 in that mode. Neither mode runs native
trainers or updates qualification records.

Filesystem metadata reads are bounded regular-file reads: 64 KiB for METADATA or
PKG-INFO and 8 KiB for direct_url.json. Named pipes, devices, symlinks, repeated
identity fields, ambiguous distributions and malformed VCS identity stay unknown.
ZIP/custom distribution providers are not modeled by this bounded reader.
The context has no public arbitrary-path inspection parameter and performs no
network requests. Its portable tests use explicitly generated dist-info fixtures,
including a named-pipe liveness control; they are not real native installations.
