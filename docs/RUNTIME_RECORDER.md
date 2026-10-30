# Record an existing inference loop at the software interfaces

`RuntimeRecorder` wraps the caller's existing synchronous `send_action` and
`get_observation` calls. It records the actual request, the original method's
returned command, selected observed state and camera metadata. It preserves the
original arguments and return objects and never retries a call. Optional
`record_policy_command` records an explicitly selected policy-output boundary,
so a changed software command can be traced to its exact request/return events.

This provides an application adapter over [FlightRecorder](FLIGHT_RECORDER.md).
It does not connect, configure, calibrate, stop or autonomously operate a robot.
The application remains responsible for connection lifetime and its existing
exception/stop behavior. There is no MCP tool for dispatching these calls.

## Attach to the existing application

The interface signatures match the genuine upstream
[LeRobot Robot API at `5831498`](https://github.com/huggingface/lerobot/blob/5831498454e93c71d209ff2e5e823e434f1efd15/src/lerobot/robots/robot.py):
`send_action(action)` returns a flat action mapping, potentially changed by the
driver, and `get_observation()` returns a flat observation mapping. The source
Git blob is `fa61ce9cee79f3593eb2980a0742e56b9eb3b180`. Compatibility with these
two signatures does not qualify a concrete hardware implementation, RobotClient,
ROS middleware, asynchronous interface or physical feedback semantics.

```python
from roboclue import RuntimeRecorder, SignalSpec

# Replace these fields and units with the application's verified declarations.
# Use units=(None,) if the original producer's unit is unknown.
position = SignalSpec(("elbow.pos",), ("rad",), "joint_position")

with RuntimeRecorder(
    robot,                         # Existing, application-owned interface.
    "new-runtime-recording.jsonl",
    action_spec=position,
    returned_spec=position,
    observation_spec=position,
    policy_spec=position,
    camera_keys=("front",),
    provenance={"kind": "caller_runtime", "producer": "my_existing_inference_loop"},
) as recording:
    observation = recording.get_observation()
    action = existing_policy_and_processors(observation)
    policy_id = recording.record_policy_command(action)
    returned_action = recording.send_action(action, policy_command_id=policy_id)
```

`examples/runtime_recorder/record_loop.py` supplies a reusable loop function that
accepts the original robot and observation-to-action callable. Its invocation
is explicit application code. It does not load a policy or hardware driver.
If policy output is a tensor, use the application's original action-to-mapping
conversion and describe that observation point; the recorder does not guess a
coordinate mapping from tensor indices. Record before a transformation as well
only when its fields, units and semantics are explicitly known.

`SignalSpec` selects ordered flat mapping keys. It accepts finite Python or
NumPy scalar values, not arbitrary tensor/array flattening. Extra mapping
fields pass through unchanged. Units and semantics are declared separately for
requested, returned and observed signals. Neither a `.pos` suffix nor a value's
range establishes radians, degrees or physical joint identity.

## What is associated, and what stays unknown

If you are deciding whether an existing recording can answer your question,
start with [Choose a workflow](CHOOSE_A_WORKFLOW.md).

Every `send_action` invocation records the request before calling the original
exactly once. Its actual returned mapping is associated with that same software
call. This association supports request/return comparisons; it does not prove
that a motor received or executed that value. The returned mapping is also
recorded as a separate software-command event, whose ID is available as
`recording.last_send.returned_command_id`.

`get_observation` defaults to `associated_command_id=None`. A later call or
neighboring row does not establish command-to-state causality. If the application
has independent evidence for an association, it may explicitly supply a returned
command ID from this recorder. Such an association remains a caller declaration.
The adapter never automatically attaches observations to the latest command.

Default timestamps use local `time.monotonic_ns` at software call/return
observation points. The observed interval includes recorder overhead and is not
device acquisition latency, exposure time, motor delivery or causal response.
Custom timestamp providers require an explicit unit and domain; unverified clock
alignment must use `domain=None`. Do not pass replay-machine wall time as the
acquisition time of old data.

For each declared camera key, the adapter records returned NumPy array dimensions
without inspecting image content. By default, its frame identity is explicitly
a **software observation ID**, not a sensor frame counter. `frame_ids` may carry
caller-supplied frame identities. Missing declared camera fields raise an
actionable input error instead of silently omitting required coverage. Camera
cadence findings describe observation-return events, not independently measured
camera exposure or fresh pixels.

Use `triage_flight_recording` with explicit criteria, or the existing RoboClue
flight-triage CLI/MCP read-only entry point, to inspect the output. Exact event
IDs identify changed request/return coordinates, units, timing or missing
associations. Clipping or unit conversion may be intentional; a residual warning
is an observed software difference, not a diagnosis of a driver or robot defect.

Residual findings keep the API code `tracking` and the original numeric criteria.
Where the recorded event kinds and association metadata identify this adapter's
comparison, `details.comparison_boundary` distinguishes `policy_to_request`
(selected policy output → software request) from `request_to_return` (request →
actual software return). Messages name that software boundary. Event IDs, stream
names or neighboring timestamps alone never establish it. Legacy or arbitrary
FlightRecorder records retain their original, unspecified comparison scope.

`roboclue.flight_view(report)` provides a compact view of a
complete triage report: selected software comparisons, all software comparison
counts, omitted-row count, and a separate physical-feedback UNKNOWN with reasons
and unassociated feedback citations. A selected comparison PASS means those
declared values met that tolerance; overall UNKNOWN means other checks lack
evidence. Even an explicitly caller-associated observation does not independently
verify acquisition timing, motor delivery or a physical response. The compact
view leaves the full report, counts and retained criterion unchanged.

## Exceptions and bounds

Native method exceptions propagate with their original identity. Schema-2
FlightRecorder finalizes the logged prefix with interruption/coverage UNKNOWN
and a bounded exception type, without copying its message or traceback. Recording
errors also propagate. An error **after** the original `send_action` ran does
not establish that no command was dispatched: do not blindly retry it.

Only explicit calls are intercepted. The proxy does not monkeypatch the original
robot, change its connection state, or capture calls made directly on that
original object. Context creation and closing invoke no robot method. Concurrent
writers, async calls and automatic background recording are unsupported. Existing
FlightRecorder event/byte limits apply; there is no rotation or OS-crash recovery.

## Public artifact and controlled software qualification

The portable replay requires a local LeRobot-format export of the public
[`nvidia/Arena-G1-Static-PickNPlace-Task`](https://huggingface.co/datasets/nvidia/Arena-G1-Static-PickNPlace-Task/tree/37ba80a99486a4c308477854f54b4e82e77777dc)
recording and optional matching episode video:

```sh
python examples/runtime_recorder/replay_lerobot.py \
  --dataset /path/to/export \
  --video /path/to/export/videos/chunk-000/observation.images.ego_view/episode_000000.mp4 \
  --output new-runtime-replay --rows 32
```

PyArrow is required for this example; PyAV is needed only when `--video` is used.
The runtime adapter itself needs neither LeRobot nor these replay dependencies.
This specific fixture uses the export's 50-FPS recorded timeline. It reads actual
43-coordinate action/state rows and decodes original video arrays when provided.

The five arms are healthy identity-preserving software returns, one deliberately
corrupted return coordinate, its correction under the same retained criteria,
unknown units/clock alignment and a producer exception. `stored_action_value`
denotes the serialized value used for a software identity comparison; it does
not assert radians or any physical unit. The nominal recorded timeline is
declared explicitly and is not measured execution duration. Its zero per-call
timestamp differences are true by construction, not a latency benchmark.

The injected return change is a controlled replay-interface fault, not a found
upstream defect or actual motor action. Observed state is never implicitly paired
with a command, so physical tracking and task success stay UNKNOWN even in the
healthy software arm. The fixture qualifies the recording/triage adapter on public
data, not a native trained-policy run, physical robot experiment, hardware-driver
certification or independent practitioner benefit.

The retained 32-row qualification decoded 32 real video frames. Healthy,
faulty, corrected and unknown arms each logged 225 events; the exception arm
retained 57 events after eight original calls. The injected return difference
was located at `send-request:51` → `send-return:53`, with one coordinate differing
by 1.0. The correction removes that warning under unchanged criterion
`4558c26892a10f8a10823be1dac87d5103c54536743ed3a091706c9deb38a1bb`.
Healthy/corrected arms still contain 96 UNKNOWN physical-association findings;
the adapter does not label them fully healthy robot runs.

The combined adapter/existing-flight suite passed 62 cases on Python 3.11.5
and 3.12.15, with Ruff passing. A separate standard-library/PyArrow audit
recomputed all recorded source values, the controlled return difference,
request/return associations, null state associations and schema-2 hash seals
without importing RoboClue, LeRobot or Torch. These are software qualification
results on public simulation data, with the limits above.
