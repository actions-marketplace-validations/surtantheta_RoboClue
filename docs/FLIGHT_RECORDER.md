# Deployment flight recorder and offline triage

`policy_doctor.flight_recorder` records what a caller observes at its actual
software command and feedback observation points. It does not invoke a policy,
send a command, open a camera, or install a hardware adapter. Recording is opt-in
and writes one bounded JSONL file to an explicitly chosen new path. Images and
unselected configuration fields are not collected.

The caller declares provenance, clock units and epochs, ordered joint identities,
per-joint units, signal semantics, command associations, camera frame IDs and
interface metadata. These are declarations, not authenticated hardware evidence.
An observed command is evidence of the software value at that observation point;
it does not establish motor delivery, movement, safety or task success.

## Instrument the existing producer

Use the existing native producer's actual values at the point where they are
observed. For a live producer, `timestamp` should be the timestamp of the stated
observation point; capture it in the caller. Do not timestamp old recorded values
with the replay machine's current clock or fabricate feedback from the command.

```python
from roboclue import FlightRecorder, triage_flight_recording

with FlightRecorder(
    "deployment-flight.jsonl",
    clocks={"host": {"unit": "s", "domain": "host_monotonic_epoch"}},
    provenance={"kind": "caller_runtime", "producer": "my_existing_policy_loop"},
) as flight:
    flight.interface(
        "interface:0", timestamp=interface_observed_at, clock="host",
        stream="existing_software_consumer",
        metadata={"driver_revision": driver_revision, "mode": "position"},
    )
    # Place this call at the producer's actual software command observation point.
    flight.command(
        "command:0", timestamp=command_observed_at, clock="host",
        stream="existing_software_consumer",
        joint_names=["elbow"], units=["rad"], values=command_values,
        semantics="joint_position",
    )
    # Record actual observed state, if available. Never copy the command here.
    flight.feedback(
        "feedback:0", command_id="command:0", timestamp=feedback_observed_at,
        clock="host", stream="existing_state_reader",
        joint_names=["elbow"], units=["rad"], values=observed_state,
        semantics="joint_position",
    )
    flight.camera(
        "camera:0", timestamp=frame_observed_at, clock="host", stream="front",
        frame_id=frame_id, metadata={"width": 640, "height": 480},
    )

report = triage_flight_recording(
    "deployment-flight.jsonl",
    max_command_gap_s=0.1, max_feedback_gap_s=0.1, max_camera_gap_s=0.1,
    max_latency_s=0.05,
    tracking_tolerance={"elbow": {"unit": "rad", "absolute": 0.02}},
)
```

The snippet names variables that must come from the caller's existing producer;
it is not a robot adapter. Call recording methods repeatedly with unique event
IDs. Use `command_id=None` when association is unobserved; row ordinal or nearest
timestamp alone does not establish association. Use `units=[None, ...]` when units
are unknown, `timestamp=None` or `clock=None` when unavailable, and a clock's
`domain=None` when its epoch/synchronization is unknown.

Clocks support `s`, `ms`, `us` and `ns`. Sharing a non-null `domain` explicitly
declares that clocks use the same aligned epoch after conversion to seconds.
Matching units or matching clock names alone does not establish synchronization.
Clock offsets are not estimated or corrected. A device clock with unverified
alignment must use its own domain or `None`; receipt time and acquisition time
should not be given the same meaning without evidence.

## What triage checks

The CLI accepts either a JSON file path or a shell-quoted inline JSON object for
`--tracking-tolerance`. Both forms use the same strict JSON reader and the same
unit-bearing tolerance checks as the Python API:

```sh
# tolerances.json contains {"elbow": {"unit": "rad", "absolute": 0.02}}
roboclue advanced flight-triage deployment-flight.jsonl --tracking-tolerance tolerances.json --json
roboclue advanced flight-triage deployment-flight.jsonl \
  --tracking-tolerance '{"elbow": {"unit": "rad", "absolute": 0.02}}' --json
```

Use the recording's declared joint names and units; the example does not infer
radians from coordinate names. Files and inline objects are limited to 4 MiB;
duplicate keys, nonfinite numbers and malformed JSON are refused. File inputs
retain the regular-file, symlink and path restrictions. Tolerances must have an
explicit unit and a finite nonnegative numeric `absolute` value, never a Boolean.

Every finding has a code, `pass`/`warning`/`unknown` status, exact event IDs,
details and an actionable remedy when a gate fails or a criterion is exceeded.
The recording path and SHA-256 bind the report to the inspected bytes; the hash
does not authenticate the producer. The caller selects diagnostic thresholds;
they are not robot safety limits.

| Check | Evidence required | Result when evidence is missing |
| --- | --- | --- |
| Command, feedback and camera timing | Two consecutive events in each declared stream, aligned timestamps and an explicit maximum gap | `unknown` |
| Observed latency | Exactly one explicitly associated feedback sample per command, comparable nonnegative timestamp difference and `max_latency_s` | `unknown` |
| Ordered joint identity | Exact ordered declared command/feedback names | A mismatch is a warning; no automatic permutation |
| Units | Explicit matching per-coordinate units | Missing units are `unknown`; mismatches are warnings; no implicit conversion |
| Signal semantics | Matching declared quantities | A mismatch is a warning |
| Command/observation residual | Explicit association, aligned nonnegative times, matching identities/units/semantics, fresh feedback if a latency bound was supplied, and per-joint unit-bearing tolerances | `unknown` |

Repeated or backwards timestamps receive timing warnings. Large timestamp gaps
identify exact adjacent events; they do not prove a scheduler or camera fault.
Large latency identifies an associated timestamp difference; it does not measure
motor delivery or establish a causal response. A residual warning names possible
inspection points such as mapping, units, clipping and timing; it does not
diagnose a mechanical cause. Duplicate associated feedback samples stay unknown
rather than selecting a convenient sample. Missing feedback remains unknown even
when command and camera timing are healthy.

Overall status is `warning` when a diagnostic warning exists, otherwise `unknown`
when any required check lacks evidence, otherwise `pass`. Healthy individual
checks retain their pass status alongside unknown checks. One event cannot prove
cadence; absent camera or feedback streams remain unknown. No threshold means
that criterion remains unknown.

For RuntimeRecorder's explicit software associations, residual findings identify
`details.comparison_boundary` as `policy_to_request` or `request_to_return` and
name that software comparison in their message. The `tracking` code, tolerance,
numeric verdict and explicit association gates remain unchanged. Arbitrary or
legacy records are never classified from stream names, event IDs or timestamps.
`roboclue.flight_view(report, max_rows=4)` returns a bounded
view of a complete report with software comparisons beside physical-feedback
UNKNOWN and its missing-association evidence. Its software PASS rows describe
selected value comparisons, while overall UNKNOWN still reflects missing checks.
This view supplies no new physical verdict and does not replace the full report.

## Relationship to existing evidence

`command_capture` and `command_path` verify a pinned native synchronous software
selection/processor/interpolation/consumer chain. Keep those witnesses when that
scope is needed. This recorder adds explicit observation-time and feedback
evidence; it does not replace the native chain's identity checks or infer time
alignment from its tick ordinal. `commanded_reach` already inspects recorded
action/state residual patterns in datasets; use it for that broader recording
analysis. `units_manifest` can inspect supported dataset unit evidence. This
recorder requires units to be declared explicitly and does not silently promote
inferred units or coordinate names to authenticated robot identity.

## Public replay qualification

The release validation replays 64 actual rows from the existing LeRobot-format
export of `nvidia/Arena-G1-Static-PickNPlace-Task` at revision
`37ba80a99486a4c308477854f54b4e82e77777dc`, through `FlightRecorder` itself. It
records the 43 named action/state coordinates, original dataset timestamps,
frame and episode IDs, source hashes, camera feature identity and interface
metadata. The resulting JSONL has 193 events including the interface event.

This is a **public recorded simulation replay**. Dataset metadata supplies
coordinate names and a nominal 50 FPS, but does not establish units, acquisition
clock synchronization, command-to-state association or motor delivery. Replay
therefore uses null units, a null clock domain and null feedback command IDs.
Timing, latency and tracking correctly remain unknown with no invented faults.
No image pixels or physical robot outcomes are claimed. A replay does not turn
recorded actions into observed policy execution on a robot.

A separate cadence control traces the retained Arena converter's timestamp
recipe: `convert_hdf5_to_lerobot.py:395` writes float64 frame index multiplied by
`1 / config.fps`; the retained configuration declares 50 FPS. It declares seconds
on that nominal per-episode recording timeline, using the original stored
timestamps unchanged. This is not an acquisition or hardware clock. A fixed
0.03-second maximum gap gives 189 passing command/state/frame-reference timing
checks. Deliberately omitting camera metadata event 32 produces one warning at
`camera:31` → `camera:33` (about 0.04 seconds); restoring the original event returns
all timing checks to pass under the same criterion. This is a controlled missing
replay event, not a claimed real camera or recording fault. Latency and physical
tracking remain unknown in all three arms, and the original all-UNKNOWN replay
remains byte-identical. Index-derived nominal cadence passes by construction;
this control checks event continuity rather than independently measured sampling.

The companion tests supply controlled software observations with explicit clocks,
units and associations. Healthy controls pass; separate residual, joint-name,
unit, semantics, latency and timing-gap faults identify exact events; corrected
controls pass again. Unknown controls cover absent feedback, unaligned epochs,
missing units/timestamps/associations, stale feedback and missing criteria.

## Bounds and incomplete captures

Limits are 10,000 events, 256 coordinates per vector, 32 clocks, 32 scalar
metadata entries per event, 128 KiB per JSONL line and 32 MiB per recording.
Strings are bounded printable text. Numeric values must be finite. Unsupported
event shapes, duplicate IDs/JSON keys, changed bytes, invalid tolerances,
symlink inputs and incomplete/hash-mismatched recordings are refused.

The writer creates its output exclusively; it refuses an existing path, FIFO,
symlink or missing parent. It flushes each event to the file stream. New recordings
use schema 2: the count/hash footer binds termination as `normal` or
`producer_exception`. A handled producer/Python exception finalizes the logged
prefix, records only its bounded exception **type**, and propagates the original
exception unchanged. Its message, traceback and local variables are not recorded.
Triage adds explicit interruption/coverage UNKNOWN and retains earlier warnings;
the prefix can never become a green full-run result. Use the original producer
traceback and cited last events to investigate; the type alone does not establish
why the producer failed.

If footer writing or closing fails, finalization is unsuccessful. Cleanup removes
the completed footer from the original output inode where the filesystem permits,
so triage refuses that file. The original body exception still propagates, with
bounded failure-type notes; cleanup cannot replace it. A failed cleanup note means
the file must be treated as unusable even if residual bytes remain. With no body
exception, the finalization error itself propagates. Legacy schema-1 clean
recordings retain their original reading and report semantics.

Actual partial, truncated, hash-mismatched or abruptly killed captures remain
refused. This release does not offer crash recovery or guaranteed disk persistence,
automatic rotation, background recording, concurrent writers, camera content
analysis, clock calibration or hardware control. Choose a bounded recording
interval and retain the producer exception separately for offline triage.
