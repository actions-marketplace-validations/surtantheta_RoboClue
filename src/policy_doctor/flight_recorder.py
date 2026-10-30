"""Opt-in, caller-instrumented software command/feedback evidence and offline triage.

No policy, adapter, network, camera, or robot is invoked. Associations, clocks,
joint identities and units are caller declarations, not authenticated facts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path

from .artifacts import InputError

MAX_BYTES = 32 * 1024 * 1024
MAX_EVENTS = 10000
MAX_LINE = 128 * 1024
_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
SCOPE = (
    "Caller-instrumented software commands and observed feedback; association, units, "
    "joint identity and synchronized clock domains are caller declarations. Latency is "
    "the observed timestamp difference, not motor delivery or causal response time. "
    "Residuals are triage evidence, not mechanical diagnoses or task success. Camera "
    "events contain metadata only; image content, hardware execution and safety remain unknown."
)


def _text(value, label):
    if (type(value) is not str or not 1 <= len(value.encode("utf8")) <= 256
            or not value.isprintable() or value != value.strip()):
        raise InputError(f"{label} requires bounded nonempty printable text")
    return value


def _number(value, label):
    if type(value) not in (int, float):
        raise InputError(f"{label} requires a finite number")
    try:
        valid = math.isfinite(value) and abs(value) <= 1e100
    except OverflowError:
        valid = False
    if not valid:
        raise InputError(f"{label} requires a bounded finite number")
    return value


def _metadata(value):
    if type(value) is not dict or len(value) > 32:
        raise InputError("Metadata requires at most 32 scalar entries")
    for key, item in value.items():
        _text(key, "Metadata key")
        if type(item) is str:
            _text(item, "Metadata value")
        elif type(item) in (int, float):
            _number(item, "Metadata value")
        elif item is not None and type(item) is not bool:
            raise InputError("Metadata values must be text, finite numbers, Boolean or null")
    return dict(value)


def _clocks(value):
    if type(value) is not dict or not 1 <= len(value) <= 32:
        raise InputError("Declare 1..32 clocks, including unknown domains explicitly")
    result = {}
    for name, definition in value.items():
        _text(name, "Clock name")
        if (type(definition) is not dict or set(definition) != {"unit", "domain"}
                or type(definition["unit"]) is not str or definition["unit"] not in _SCALE):
            raise InputError("Each clock requires unit s/ms/us/ns and domain (text or null)")
        if definition["domain"] is not None:
            _text(definition["domain"], "Aligned clock domain")
        result[name] = dict(definition)
    return result


def _vector(names, units, values, semantics):
    if (type(names) is not list or not 1 <= len(names) <= 256
            or len(set(_text(n, "Joint name") for n in names)) != len(names)):
        raise InputError("joint_names requires 1..256 distinct declared names")
    if type(units) is not list or len(units) != len(names):
        raise InputError("Declare one unit or null per joint")
    for unit in units:
        if unit is not None:
            _text(unit, "Joint unit")
    if type(values) is not list or len(values) != len(names):
        raise InputError("Declare one finite value per joint")
    for value in values:
        _number(value, "Joint value")
    _text(semantics, "Signal semantics")
    return {"joint_names": list(names), "units": list(units), "values": list(values), "semantics": semantics}


def _event(value, clocks):
    if (type(value) is not dict or type(value.get("kind")) is not str
            or value["kind"] not in {"command", "feedback", "camera", "interface"}):
        raise InputError("Unsupported flight event")
    kind = value["kind"]
    expected = {"kind", "event_id", "timestamp", "clock", "stream", "metadata"}
    if kind in {"command", "feedback"}:
        expected |= {"joint_names", "units", "values", "semantics"}
    if kind == "feedback":
        expected.add("command_id")
    if kind == "camera":
        expected.add("frame_id")
    if set(value) != expected:
        raise InputError("Unexpected or missing flight event fields")
    _text(value["event_id"], "Event ID")
    _text(value["stream"], "Stream/interface identity")
    if value["timestamp"] is not None:
        _number(value["timestamp"], "Timestamp")
    if value["clock"] is not None and (type(value["clock"]) is not str or value["clock"] not in clocks):
        raise InputError("Event clock must reference a declared clock or null")
    result = dict(value)
    result["metadata"] = _metadata(value["metadata"])
    if kind in {"command", "feedback"}:
        result.update(_vector(value["joint_names"], value["units"], value["values"], value["semantics"]))
    if kind == "feedback" and value["command_id"] is not None:
        _text(value["command_id"], "Associated command ID")
    if kind == "camera":
        _text(value["frame_id"], "Camera frame ID")
    return result


def _dump(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf8")


def _exception_name(exception_type):
    """Bounded type identity only; never inspect an exception's message or repr."""
    try:
        module = type.__getattribute__(exception_type, "__module__")
        name = type.__getattribute__(exception_type, "__name__")
        label = module + "." + name
        if not label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_." for character in label):
            return "BaseException"
        return label[:128]
    except BaseException:
        return "BaseException"


def _termination(value):
    if (type(value) is not dict or set(value) != {"status", "exception_type"}
            or value["status"] not in ("normal", "producer_exception")):
        raise InputError("Unsupported flight recording termination")
    if value["status"] == "normal":
        if value["exception_type"] is not None:
            raise InputError("Normal termination cannot contain an exception type")
    else:
        label = value["exception_type"]
        if (type(label) is not str or not 1 <= len(label) <= 128
                or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_." for character in label)):
            raise InputError("Producer exception requires a bounded type identity only")
    return dict(value)


def _note(exception, message):
    # A user exception may override add_note. Never let diagnostics replace it.
    try:
        BaseException.add_note(exception, message)
    except BaseException:
        pass


class FlightRecorder:
    """Write an exclusive, bounded JSONL recording, closed by a hash/count footer.

    Record immediately at the caller's actual command/feedback observation points.
    No clock sampling or command dispatch is performed by this class. Concurrent
    writers are unsupported. A handled producer exception finalizes a bounded
    prefix with explicit interruption coverage and propagates unchanged. Abrupt
    termination and unverifiable writes remain unsupported, not crash recovery.
    """

    def __init__(self, path, *, clocks, provenance, max_events=MAX_EVENTS):
        self.path = Path(path)
        self.clocks = _clocks(clocks)
        self.provenance = _metadata(provenance)
        if not self.provenance:
            raise InputError("Explicit recording provenance is required")
        if type(max_events) is not int or not 1 <= max_events <= MAX_EVENTS:
            raise InputError(f"max_events must be in 1..{MAX_EVENTS}")
        self.max_events = max_events
        self._stream = None
        self._used = False
        self._ids = set()
        self._bytes = 0
        self._digest = hashlib.sha256()
        self._identity = None

    def __enter__(self):
        if self._used:
            raise InputError("FlightRecorder cannot be reused")
        self._used = True
        try:
            self._stream = self.path.open("xb")
        except OSError as exc:
            raise InputError(f"Cannot create exclusive flight recording: {exc}") from exc
        identity = os.fstat(self._stream.fileno())
        self._identity = (identity.st_dev, identity.st_ino)
        self._write({"kind": "flight_recording", "schema_version": 2, "clocks": self.clocks,
                     "provenance": self.provenance})
        return self

    def _write(self, value, *, footer=False):
        raw = _dump(value)
        if len(raw) > MAX_LINE or self._bytes + len(raw) > MAX_BYTES - (0 if footer else 1024):
            raise InputError("Flight recording exceeds its byte budget")
        self._stream.write(raw)
        self._stream.flush()
        self._bytes += len(raw)
        if not footer:
            self._digest.update(raw)

    def _record(self, kind, event_id, timestamp, clock, stream, metadata, **fields):
        if self._stream is None or self._stream.closed:
            raise InputError("Use FlightRecorder inside its context manager")
        value = _event({"kind": kind, "event_id": event_id, "timestamp": timestamp,
                        "clock": clock, "stream": stream, "metadata": {} if metadata is None else metadata,
                        **fields}, self.clocks)
        if event_id in self._ids or len(self._ids) >= self.max_events:
            raise InputError("Duplicate flight event ID or event count budget exceeded")
        self._write(value)
        self._ids.add(event_id)
        return event_id

    def command(self, event_id, *, timestamp, clock, stream, joint_names, units, values, semantics, metadata=None):
        """Record the observed policy/software command; this never sends it."""
        return self._record("command", event_id, timestamp, clock, stream, metadata,
                            **_vector(joint_names, units, values, semantics))

    def feedback(self, event_id, *, command_id, timestamp, clock, stream, joint_names, units, values,
                 semantics, metadata=None):
        """Record observed state and an explicit association (null if unknown)."""
        return self._record("feedback", event_id, timestamp, clock, stream, metadata, command_id=command_id,
                            **_vector(joint_names, units, values, semantics))

    def camera(self, event_id, *, timestamp, clock, stream, frame_id, metadata=None):
        """Record frame/time/source metadata; images are not collected."""
        return self._record("camera", event_id, timestamp, clock, stream, metadata, frame_id=frame_id)

    def interface(self, event_id, *, timestamp, clock, stream, metadata):
        """Record caller-selected adapter/interface configuration or observations."""
        return self._record("interface", event_id, timestamp, clock, stream, metadata)

    def __exit__(self, exc_type, exc, tb):
        footer_start = self._bytes
        failures = []
        try:
            termination = {"status": "normal" if exc_type is None else "producer_exception",
                           "exception_type": None if exc_type is None else _exception_name(exc_type)}
            seal = {"events": len(self._ids), "termination": termination}
            digest = self._digest.copy()
            digest.update(_dump(seal))
            self._write({"kind": "flight_recording_end", **seal, "sha256": digest.hexdigest()}, footer=True)
        except BaseException as error:
            failures.append(error)
        try:
            self._stream.close()
        except BaseException as error:
            failures.append(error)
        if failures:
            invalidated = self._invalidate_footer(footer_start)
            original = exc if exc is not None else failures[0]
            for error in failures:
                _note(original, "FlightRecorder finalization failed (" + _exception_name(type(error))
                      + "); recording is unusable. "
                      + ("Completed footer removed." if invalidated else "Footer invalidation also failed; do not triage this file."))
            if exc is None:
                raise original
        return False

    def _invalidate_footer(self, offset):
        """Remove even a fully written footer after finalization/close failure.

        Only truncate the original newly created inode; never a replacement path.
        If the filesystem itself refuses cleanup, the propagated error/note is
        authoritative and no successful finalization is reported.
        """
        descriptor = None
        try:
            flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(self.path, flags)
            identity = os.fstat(descriptor)
            if not stat.S_ISREG(identity.st_mode) or (identity.st_dev, identity.st_ino) != self._identity:
                return False
            os.ftruncate(descriptor, offset)
            return True
        except BaseException:
            return False
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError("Duplicate JSON key in flight recording")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InputError("Invalid flight recording JSON") from exc


def _load(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise InputError("Flight recording must be a regular file, not a symlink")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise InputError("Flight recording byte budget exceeded")
    lines = raw.splitlines(keepends=True)
    if len(lines) < 2 or len(lines) > MAX_EVENTS + 2 or any(len(line) > MAX_LINE for line in lines):
        raise InputError("Incomplete or oversized flight recording")
    header, footer = _parse(lines[0]), _parse(lines[-1])
    if (type(header) is not dict or set(header) != {"kind", "schema_version", "clocks", "provenance"}
            or header["kind"] != "flight_recording" or type(header["schema_version"]) is not int
            or header["schema_version"] not in (1, 2)):
        raise InputError("Unsupported flight recording header")
    clocks = _clocks(header["clocks"])
    if not _metadata(header["provenance"]):
        raise InputError("Explicit recording provenance is required")
    footer_keys = {"kind", "events", "sha256"} | ({"termination"} if header["schema_version"] == 2 else set())
    if (type(footer) is not dict or set(footer) != footer_keys
            or footer["kind"] != "flight_recording_end" or type(footer["events"]) is not int
            or footer["events"] != len(lines) - 2):
        raise InputError("Missing flight recording footer or event count/hash mismatch")
    termination = _termination(footer["termination"]) if header["schema_version"] == 2 else None
    digest = hashlib.sha256(b"".join(lines[:-1]))
    if termination is not None:
        digest.update(_dump({"events": footer["events"], "termination": termination}))
    if footer["sha256"] != digest.hexdigest():
        raise InputError("Missing flight recording footer or event count/hash mismatch")
    events = [_event(_parse(line), clocks) for line in lines[1:-1]]
    if len({event["event_id"] for event in events}) != len(events):
        raise InputError("Duplicate flight event IDs")
    return header, events, hashlib.sha256(raw).hexdigest(), termination


def _elapsed(left, right, clocks):
    if left["timestamp"] is None or right["timestamp"] is None or left["clock"] is None or right["clock"] is None:
        return None
    a, b = clocks[left["clock"]], clocks[right["clock"]]
    if a["domain"] is None or a["domain"] != b["domain"]:
        return None
    return right["timestamp"] * _SCALE[b["unit"]] - left["timestamp"] * _SCALE[a["unit"]]


def _comparison_boundary(command, feedback):
    """Classify explicit recorder declarations, never IDs, streams or timestamps."""
    if command["kind"] != "command" or feedback["kind"] != "feedback":
        return None
    if feedback["command_id"] != command["event_id"]:
        return None
    point = command["metadata"].get("observation_point")
    association = feedback["metadata"].get("association")
    if (point == "caller_policy_output"
            and association == "explicit_caller_policy_to_request_not_physical_feedback"):
        return "policy_to_request"
    if (point == "before_original_send_action"
            and association == "same_original_send_action_call_return"
            and feedback["metadata"].get("physical_delivery_verified") is False):
        return "request_to_return"
    return None


def triage_flight_recording(path, *, max_command_gap_s=None, max_feedback_gap_s=None,
                            max_camera_gap_s=None, max_latency_s=None, tracking_tolerance=None):
    """Inspect a completed recording using caller-selected thresholds.

    ``tracking_tolerance`` is ``{joint_name: {"unit": unit, "absolute": number}}``.
    It applies only to exact ordered names, units and semantics with an explicit
    command association and comparable, nonnegative timestamps. One associated
    feedback per command is required; duplicates stay UNKNOWN rather than choosing
    a convenient sample. Thresholds are diagnostic criteria, never safety limits.
    """
    limits = {"command": max_command_gap_s, "feedback": max_feedback_gap_s, "camera": max_camera_gap_s}
    for value in [*limits.values(), max_latency_s]:
        if value is not None and _number(value, "Time threshold") <= 0:
            raise InputError("Time thresholds must be positive seconds")
    if tracking_tolerance is not None:
        if type(tracking_tolerance) is not dict or not 1 <= len(tracking_tolerance) <= 256:
            raise InputError("tracking_tolerance requires a bounded joint/unit/absolute mapping")
        for name, limit in tracking_tolerance.items():
            _text(name, "Tolerance joint")
            if type(limit) is not dict or set(limit) != {"unit", "absolute"}:
                raise InputError("Tracking tolerance requires explicit unit and absolute")
            _text(limit["unit"], "Tolerance unit")
            if _number(limit["absolute"], "Absolute tolerance") < 0:
                raise InputError("Absolute tolerance cannot be negative")
    header, events, digest, termination = _load(path)
    findings = []

    def add(code, status, message, selected=(), remedy=None, **details):
        if code == "tracking" and len(selected) == 2:
            boundary = _comparison_boundary(*selected)
            if boundary is not None:
                details["comparison_boundary"] = boundary
                if "absolute_residual" in details:
                    label = {"policy_to_request": "policy output → send request",
                             "request_to_return": "send request → software return"}[boundary]
                    message = f"Software {label} absolute residual; physical execution unverified."
                    if status == "warning":
                        remedy = "Inspect the selected software values, transformations, mapping and declared units at these event IDs."
        findings.append({"code": code, "status": status, "message": message, "remedy": remedy,
                         "event_ids": [event["event_id"] for event in selected], "details": details})

    if termination is not None and termination["status"] == "producer_exception":
        add("recording_interruption", "unknown", "Producer exited by exception; checks cover only the logged prefix.",
            events[-1:], "Inspect the original producer traceback and last events, then rerun the intended interval; "
            "exception type alone does not establish cause.", exception_type=termination["exception_type"],
            logged_events=len(events), requested_run_coverage="unknown")
    clocks = header["clocks"]
    for kind, limit in limits.items():
        streams = {}
        for event in events:
            if event["kind"] == kind:
                streams.setdefault(event["stream"], []).append(event)
        if not streams:
            add(f"{kind}_timing", "unknown", f"No {kind} events recorded.",
                remedy=f"Instrument actual {kind} observations and their clock declarations.")
        for stream, samples in streams.items():
            if len(samples) < 2 or limit is None:
                add(f"{kind}_timing", "unknown", "Need two events and an explicit gap threshold.", samples,
                    "Record consecutive observations and declare an expected maximum gap in seconds.", stream=stream)
                continue
            for left, right in zip(samples, samples[1:], strict=False):
                gap = _elapsed(left, right, clocks)
                if gap is None:
                    add(f"{kind}_timing", "unknown", "Clock synchronization or timestamp evidence is missing.",
                        [left, right], "Declare aligned clock epochs only after verifying synchronization.", stream=stream)
                elif gap <= 0:
                    add(f"{kind}_timing", "warning", "Recorded timestamps are repeated or move backwards.",
                        [left, right], "Check timestamp origin, units, duplicate frames and recording order.", gap_s=gap, stream=stream)
                else:
                    add(f"{kind}_timing", "warning" if gap > limit else "pass",
                        "Observed timestamp gap exceeds criterion." if gap > limit else "Observed timestamp gap meets criterion.",
                        [left, right], "Inspect scheduling, buffering and timestamp capture at these events." if gap > limit else None,
                        gap_s=gap, threshold_s=limit, stream=stream)
    commands = {event["event_id"]: event for event in events if event["kind"] == "command"}
    associations = {}
    for event in events:
        if event["kind"] != "feedback":
            continue
        if event["command_id"] not in commands:
            observation = (event["metadata"].get("observation_point")
                           == "software_return_not_sensor_acquisition")
            add("feedback_association", "unknown", "Feedback has no associated recorded command.", [event],
                "Carry the actual observed command ID through the producer; do not pair by row ordinal.",
                **({"observation_boundary": "observation_return"} if observation else {}))
        else:
            associations.setdefault(event["command_id"], []).append(event)
    if not commands:
        add("tracking", "unknown", "No commands recorded.", remedy="Instrument the actual software command observation point.")
        add("latency", "unknown", "No commands recorded.")
    for command_id, command in commands.items():
        feedbacks = associations.get(command_id, [])
        if len(feedbacks) != 1:
            for code in ("latency", "tracking"):
                add(code, "unknown", "Exactly one explicitly associated feedback observation is required.",
                    [command, *feedbacks], "Record an unambiguous associated feedback sample; inspect missing or duplicate IDs.")
            continue
        feedback = feedbacks[0]
        pair = [command, feedback]
        elapsed = _elapsed(command, feedback, clocks)
        comparable = elapsed is not None and elapsed >= 0
        if not comparable or max_latency_s is None:
            add("latency", "unknown", "Aligned timestamps and an explicit latency criterion are required.", pair,
                "Verify clock domain/epoch/units and association, then set max_latency_s.", observed_delta_s=elapsed)
        else:
            slow = elapsed > max_latency_s
            add("latency", "warning" if slow else "pass", "Associated observed timestamp difference.", pair,
                "Inspect producer, buffers, acquisition timestamps and clock alignment at these events." if slow else None,
                observed_delta_s=elapsed, threshold_s=max_latency_s)
        names_match = command["joint_names"] == feedback["joint_names"]
        units_known = all(unit is not None for unit in command["units"] + feedback["units"])
        units_match = command["units"] == feedback["units"]
        semantics_match = command["semantics"] == feedback["semantics"]
        add("joint_identity", "pass" if names_match else "warning", "Declared ordered joint identities compared.", pair,
            None if names_match else "Check producer joint ordering and explicit mapping before comparing values.",
            command_names=command["joint_names"], feedback_names=feedback["joint_names"])
        add("units", "unknown" if not units_known else "pass" if units_match else "warning",
            "Declared coordinate units compared; no units are inferred from values.", pair,
            None if units_known and units_match else "Verify original producer and consumer units; declare or correct them explicitly.",
            command_units=command["units"], feedback_units=feedback["units"])
        add("signal_semantics", "pass" if semantics_match else "warning", "Declared signal semantics compared.", pair,
            None if semantics_match else "Compare equivalent quantities (for example position to position).")
        stale = max_latency_s is not None and elapsed is not None and elapsed > max_latency_s
        if not (comparable and names_match and units_known and units_match and semantics_match) or stale:
            add("tracking", "unknown", "Association, time, identity, unit or semantics gate prevents residual comparison.", pair,
                "Resolve the accompanying clock, latency, identity, units or semantics finding before comparing values.")
            continue
        if tracking_tolerance is None:
            add("tracking", "unknown", "No explicit per-joint unit-bearing residual tolerance supplied.", pair,
                "Declare tracking_tolerance for these joint names and units.")
            continue
        for name, unit, target, observed in zip(command["joint_names"], command["units"], command["values"],
                                               feedback["values"], strict=True):
            limit = tracking_tolerance.get(name)
            if limit is None or limit["unit"] != unit:
                add("tracking", "unknown", "No matching unit-bearing tolerance for this joint.", pair,
                    "Supply a tolerance in the declared signal unit for this exact joint.", joint=name, unit=unit)
                continue
            residual = abs(observed - target)
            bad = residual > limit["absolute"]
            add("tracking", "warning" if bad else "pass", "Associated command/observation absolute residual.", pair,
                "Inspect joint mapping, units, acquisition timing, clipping and controller logs; this does not identify a mechanical cause." if bad else None,
                joint=name, unit=unit, command=target, observed=observed, absolute_residual=residual,
                threshold=limit["absolute"])
    counts = {status: sum(finding["status"] == status for finding in findings) for status in ("pass", "warning", "unknown")}
    report = {"kind": "flight_recording_triage", "schema_version": header["schema_version"], "scope": SCOPE,
            "status": "warning" if counts["warning"] else "unknown" if counts["unknown"] else "pass",
            "recording": {"path": str(path), "sha256": digest, "events": len(events),
                          "provenance": header["provenance"], "clocks": clocks},
            "criteria": {"max_command_gap_s": max_command_gap_s, "max_feedback_gap_s": max_feedback_gap_s,
                         "max_camera_gap_s": max_camera_gap_s, "max_latency_s": max_latency_s,
                         "tracking_tolerance": tracking_tolerance},
            "counts": counts, "findings": findings}
    if termination is not None:
        report["termination"] = termination
        report["coverage"] = {"logged_events": len(events), "producer_termination": termination["status"],
                              "requested_run_coverage": "unknown" if termination["status"] == "producer_exception" else "not_asserted"}
        if termination["status"] == "producer_exception":
            report["scope"] += " Producer exception interrupted recording; requested-run coverage is UNKNOWN and only the logged prefix is assessed."
    return report
