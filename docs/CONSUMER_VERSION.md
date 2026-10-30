# Compare with the LeRobot version you actually use

The Python environment running RoboClue may differ from the environment running
your robot. Set `ROLLOUT_COMMAND` to the real command you intend to use, then
supply that environment's version:

```sh
roboclue check DATASET --lerobot-version 0.6.1 \
  --robot-command "$ROLLOUT_COMMAND" --require deployment-units --json
```

The quoted command is read as configuration, never executed. It is a caller's
declaration; a matching version label does not verify the files installed on a
remote robot. The result compares supported units and camera names, rather
than approving the whole policy, calibration or robot behavior.

## Tested source profiles

| LeRobot source | Supported robot configurations | What was checked |
| --- | --- | --- |
| Published `0.6.1` wheel | SO-100, SO-101, Koch, OMX, bimanual SO, LeKiwi arm | Original configuration construction, body/gripper normalization functions and argument parsing |
| Recorded `0.6.2` source build at `146d847` | Same six configurations | The same native source checks |

The published wheel was obtained from PyPI and checked against its declared
SHA256 `1894516040c65f80a45bd9741f8174aae90ed5d93da0627ab4f1a85fd8d75e90`.
It was the current published version at the 2026-10-27 check. The local `0.6.2`
build is a recorded Git source identity, not a claim that PyPI publishes 0.6.2.
The supported matrix is available as `supported_consumer_profiles()` in the
profile helper. It is not a compatibility range such as “all 0.6.x”.

Native defaults in these profiles:

- SO-100/SO-101, each bimanual SO arm and the LeKiwi arm use degrees by default.
- Koch and OMX use the -100..100 calibrated body-joint scale by default.
- `use_degrees=false` chooses -100..100 for body joints; `true` chooses degrees.
  Grippers use the 0..100 calibrated scale in the tested constructors.
- SO/Koch/OMX and bimanual camera dictionaries default to empty mappings.
  Omitting cameras is therefore compared as zero supplied cameras. A dataset
  with image views receives a warning naming the missing views.
- LeKiwi's tested factory supplies `front` and `wrist`; those names are compared
  as defaults. Base velocities, actual devices and physical behavior remain
  outside the units profile.
- In both recorded sources, `lerobot-record` collects teleoperation data;
  `lerobot-rollout` runs a policy. No robot constructor, device connection,
  policy inference or motor write was used in the source qualification.

[PyPI release](https://pypi.org/project/lerobot/0.6.1/) ·
[Recorded SO configuration](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/robots/so_follower/config_so_follower.py) ·
[Recorded normalization source](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/motors/motors_bus.py).

Exact per-family source hashes and the wheel identity ship in the package.
Only those critical source slices were qualified; dependencies, plugins, a whole
source tree and the actual robot environment were not attested. The production
helper does not import LeRobot.

## Missing or unsupported versions stay unresolved

A required comparison cannot clear when the version or a tested source profile
is missing. The Python helper supports `require_version=True`; canonical
consumer checks use this stricter boundary. Even an explicit `use_degrees` flag
cannot establish an unsupported version's semantics.

Recorded dataset-unit inference is retained. Old conditional comparison findings
are saved under `conditional_comparison`, rather than being presented as an
active match or mismatch for an unqualified version. Independent dataset errors
remain visible. Do not relabel the version or edit the dataset simply to get a
passing result.

The low-level helper's default remains compatible for existing callers that
explicitly use conditional, unversioned comparisons. Those results describe
the recorded source assumptions. They do not identify the robot environment.

Detailed metadata distinguishes the supplied version, tested source profile,
command role and `fleet_verified: false`. An unqualified command can retain a
`conditional_command_role` note describing the tested versions' behavior; that
note never qualifies the supplied environment or changes an UNKNOWN into PASS.

## Optional local source matching in Python

```python
from policy_doctor.consumer_version import apply_consumer_version

scoped_report = apply_consumer_version(
    units_report,
    "the actual version label",
    source_root="/trusted/local/lerobot-checkout",
    require_version=True,
)
```

The source reader hashes only the family's fixed five or seven files, with a
2 MiB total budget. It refuses symlinks and special files and executes no
supplied Python. An exact match can establish the recorded source semantics
under a different label; a mismatching tree overrides a familiar label and
leaves the comparison unresolved. This is an integration API, not automatic
remote environment detection.

The helper returns a copy and changes no original report, dataset or robot.
