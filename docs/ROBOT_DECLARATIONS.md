# Compare the robot declarations you actually supplied

A misspelled robot option must not disappear and expose a different default.
Canonical checks validate supported `--robot.*` keys, nested arm/camera keys and
unit/stream Boolean values before comparing the declared interface. For example,
`use_degres`, `usedegrees` and `use_degrees=flase` are invalid supported inputs
(exit2); the answer names the option. A flag from another native version may need
a separately reviewed profile, rather than being a typo in that version.

The field-name profiles cite LeRobot
[`146d847`](https://github.com/huggingface/lerobot/tree/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/robots):
base robot settings, supported SO/Koch/OMX/LeKiwi/bimanual robot configurations,
and the supported OpenCV/RealSense camera declarations. Unknown robot or camera
plugins stay explicitly unassessed. Configuration files are read as bounded
JSON/YAML mappings; no configuration code, device or policy is executed.
The units and camera-default comparisons have separately qualified source
profiles for the published 0.6.1 wheel and the recorded 0.6.2 source build.
Supply the actual consumer's `--lerobot-version`; see
[tested versions and robot families](CONSUMER_VERSION.md).

## Camera keys before a checkpoint exists

```sh
roboclue check ./dataset --lerobot-version 0.6.1 --robot-command 'lerobot-rollout --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.use_degrees=false --robot.cameras="{up: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"'
```

Keep the inner quotes around the camera mapping: RoboClue reads the command text
as a shell would split it, without running a shell. A malformed mapping or an
unresolved shell variable cannot become an assumed matching camera interface.
Use the registered whole-mapping option, such as `--robot.cameras="{...}"`;
the inspected native parser does not register dynamic
`--robot.cameras.front.width` options. Per-arm dataclass fields remain distinct
from their camera dictionaries.
A JSON/YAML `--robot-config` with the equivalent `robot.cameras` mapping follows
the same supported declaration checks.

Supply one resolved native command, rather than a shell script. Canonical checks
refuse multiple commands, pipes, background operators, redirection and command
substitution instead of silently comparing only the first command. Quoted or
escaped literal values and camera mappings remain data. Compare collection and
rollout settings separately when both matter.

The camera component compares local `meta/info.json` keys beginning with
`observation.images.` against supplied robot-camera output names or the tested
native defaults. SO/Koch/OMX and bimanual configurations default to no cameras;
LeKiwi defaults to `front` and `wrist`. A missing mapping must therefore not
hide a zero-camera comparison. A
bimanual arm's camera names receive its declared `left_`/`right_` prefix. An
explicit supported `rename_map` maps raw robot feature keys to the compared
dataset keys; mappings are never guessed from a familiar checkpoint name.

The result distinguishes:

- **Native defaults:** omitted cameras are compared using the qualified
  configuration's defaults. Zero cameras against image-bearing data produces a
  warning naming the missing views. A missing required or unsupported version leaves the default
  unresolved, rather than inventing an empty set.
- **Unassessed:** an explicit camera mapping, plugin, rename operation or local
  dataset declaration could not be established. It is not a camera-key PASS.
- **Disjoint identities:** the nonempty declared set shares no dataset camera
  identity without a matching explicit mapping. The report names the mismatch;
  it is not proof that an uninspected future policy fails.
- **Partial or empty selection:** dataset views are omitted from the declared
  input set. This produces a warning requiring review of policy intent, because
  selecting fewer views can be deliberate.
- **Additional keys:** the robot declares views absent from dataset metadata;
  these can be intentional and require review.
- **Consistent keys:** the supplied and dataset camera feature-key sets agree.

**Only keys are compared.** Image size, frame rate, joint order, actual camera
identity, calibration, decoded images and policy compatibility remain
unassessed. A policy can deliberately select a subset of views; this check does
not inspect that selection. Review intended view selection or a source-backed
owned rename/config correction, then repeat the same comparison. Do not edit a
public dataset merely to make the key sets agree.

`lerobot-record` is collection-only in both tested source profiles. Supplying its
robot settings describes an interface; it does not establish a deployed-policy
execution path. Use the actual applicable consumer command/config when available.
No recorder is started by `check` or the default MCP preflight.

## Python and MCP

The default `preflight_check` tool accepts `robot_command` or `robot_config`.
Its camera component retains whether the comparison was requested, whether it
actually ran, the named sets and its evidence. A key match must not be presented
as size/rate/physical compatibility.

Python integrations can import `validate_robot_declaration` and
`check_declared_camera_keys` from `roboclue`.
These preserve the existing diagnostic engines' original scoped reports; public
composition does not turn a units-only PASS into all-interface approval.

Input reads are bounded and respect configured MCP read roots. JSON duplicate
keys, YAML aliases/recursion, unsupported nesting, links and special-file input
are refused. Generic Hub checks do not materialize this local comparison; supply
the actual downloaded dataset. [Units](UNITS_MANIFEST.md) · [Main check](CHECK.md).
