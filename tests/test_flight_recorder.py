"""Controlled software evidence only; these tests do not execute robot commands."""

import hashlib
import json
import os

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.flight_recorder import FlightRecorder, triage_flight_recording

CLOCKS = {"host": {"unit": "s", "domain": "declared_aligned_epoch"},
          "sensor": {"unit": "ms", "domain": "declared_aligned_epoch"}}
CRITERIA = {"max_command_gap_s": .11, "max_feedback_gap_s": .11,
            "max_camera_gap_s": .11, "max_latency_s": .03,
            "tracking_tolerance": {"elbow": {"unit": "rad", "absolute": .02}}}


def vector(**changes):
    return {"joint_names": ["elbow"], "units": ["rad"], "values": [.5],
            "semantics": "joint_position", **changes}


def record(path, *, clocks=None, feedback=True, change=None, gap=.1, body_exception=None):
    with FlightRecorder(path, clocks=clocks or CLOCKS, provenance={"kind": "controlled_software_test"}) as recorder:
        recorder.interface("interface", timestamp=0., clock="host", stream="software_consumer",
                           metadata={"mode": "position", "driver": "caller_declared"})
        for i in range(2):
            t = i * gap
            recorder.command(f"c{i}", timestamp=t, clock="host", stream="software_consumer", **vector())
            recorder.camera(f"cam{i}", timestamp=t, clock="host", stream="front", frame_id=f"frame{i}",
                            metadata={"width": 640, "height": 480})
            if feedback:
                fields = {"command_id": f"c{i}", "timestamp": (t + .01) * 1000,
                          "clock": "sensor", "stream": "joint_observer", **vector()}
                if change:
                    fields.update(change)
                recorder.feedback(f"f{i}", **fields)
        if body_exception is not None:
            raise body_exception
    return path


def findings(report, code):
    return [finding for finding in report["findings"] if finding["code"] == code]


def test_healthy_complete_recording_preserves_controls(tmp_path):
    report = triage_flight_recording(record(tmp_path / "healthy.jsonl"), **CRITERIA)
    assert report["status"] == "pass"
    assert report["counts"]["warning"] == report["counts"]["unknown"] == 0
    assert report["recording"]["events"] == 7
    assert findings(report, "latency")[0]["details"]["observed_delta_s"] == .01
    assert findings(report, "tracking")[0]["details"]["joint"] == "elbow"
    assert "not motor delivery" in report["scope"]
    assert report["schema_version"] == 2
    assert report["termination"] == {"status": "normal", "exception_type": None}


@pytest.mark.parametrize("change,code", [({"values": [.8]}, "tracking"),
                                         ({"joint_names": ["wrist"]}, "joint_identity"),
                                         ({"units": ["degree"]}, "units"),
                                         ({"semantics": "joint_velocity"}, "signal_semantics"),
                                         ({"timestamp": 500.}, "latency")])
def test_fault_exact_events_and_correction(tmp_path, change, code):
    faulty = triage_flight_recording(record(tmp_path / "fault.jsonl", change=change), **CRITERIA)
    detected = [finding for finding in findings(faulty, code) if finding["status"] == "warning"]
    assert detected
    assert detected[0]["event_ids"] == ["c0", "f0"]
    assert detected[0]["remedy"]
    if code in {"joint_identity", "units", "signal_semantics", "latency"}:
        assert all(f["status"] == "unknown" for f in findings(faulty, "tracking"))
    corrected = triage_flight_recording(record(tmp_path / "corrected.jsonl"), **CRITERIA)
    assert corrected["status"] == "pass"


def test_gap_fault_and_corrected_timing(tmp_path):
    report = triage_flight_recording(record(tmp_path / "gap.jsonl", gap=.5), **CRITERIA)
    for code, ids in [("command_timing", ["c0", "c1"]), ("feedback_timing", ["f0", "f1"]),
                      ("camera_timing", ["cam0", "cam1"])]:
        finding = findings(report, code)[0]
        assert finding["status"] == "warning"
        assert finding["event_ids"] == ids
        assert finding["details"]["gap_s"] == pytest.approx(.5)
    assert all(f["status"] == "pass" for f in findings(report, "tracking"))
    assert triage_flight_recording(record(tmp_path / "fixed.jsonl"), **CRITERIA)["status"] == "pass"


def test_no_feedback_remains_unknown(tmp_path):
    report = triage_flight_recording(record(tmp_path / "missing.jsonl", feedback=False), **CRITERIA)
    for code in ("feedback_timing", "latency", "tracking"):
        assert all(f["status"] == "unknown" for f in findings(report, code))
    assert report["status"] == "unknown"
    assert findings(report, "command_timing")[0]["status"] == "pass"


@pytest.mark.parametrize("domain", [None, "different_epoch"])
def test_same_units_do_not_imply_synchronized_clocks(tmp_path, domain):
    clocks = {**CLOCKS, "sensor": {"unit": "ms", "domain": domain}}
    report = triage_flight_recording(record(tmp_path / "unsynced.jsonl", clocks=clocks), **CRITERIA)
    assert all(f["status"] == "unknown" for f in findings(report, "latency") + findings(report, "tracking"))
    assert all(f["status"] == "pass" for f in findings(report, "joint_identity"))


@pytest.mark.parametrize("change", [{"units": [None]}, {"clock": None}, {"timestamp": None},
                                    {"timestamp": -1}, {"command_id": None}])
def test_missing_declarations_or_negative_delta_prevent_tracking(tmp_path, change):
    report = triage_flight_recording(record(tmp_path / "unknown.jsonl", change=change), **CRITERIA)
    assert findings(report, "tracking")
    assert all(f["status"] == "unknown" for f in findings(report, "tracking"))


def test_unconfigured_criteria_do_not_make_false_pass(tmp_path):
    report = triage_flight_recording(record(tmp_path / "no-criteria.jsonl"))
    assert report["status"] == "unknown"
    assert all(f["status"] == "unknown" for f in findings(report, "tracking") + findings(report, "latency"))


def test_tolerance_unit_cannot_silently_compare_degrees_and_radians(tmp_path):
    report = triage_flight_recording(record(tmp_path / "unit.jsonl"),
                                     **{**CRITERIA, "tracking_tolerance": {"elbow": {"unit": "degree", "absolute": 1}}})
    assert all(f["status"] == "unknown" for f in findings(report, "tracking"))


def test_duplicate_feedback_not_cherry_picked(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    with FlightRecorder(path, clocks=CLOCKS, provenance={"kind": "controlled"}) as recorder:
        recorder.command("c", timestamp=0, clock="host", stream="policy", **vector())
        for i in range(2):
            recorder.feedback(f"f{i}", command_id="c", timestamp=.01 + i * .01,
                              clock="host", stream="observer", **vector())
    report = triage_flight_recording(path, **CRITERIA)
    assert findings(report, "tracking")[0]["status"] == "unknown"
    assert findings(report, "tracking")[0]["event_ids"] == ["c", "f0", "f1"]


def test_incomplete_and_tampered_recordings_refused(tmp_path):
    path = record(tmp_path / "record.jsonl")
    raw = path.read_bytes()
    path.write_bytes(b"".join(raw.splitlines(keepends=True)[:-1]))
    with pytest.raises(InputError):
        triage_flight_recording(path)
    path.write_bytes(raw.replace(b'"values": [0.5]', b'"values": [0.9]', 1))
    with pytest.raises(InputError, match="hash"):
        triage_flight_recording(path)


def test_handled_producer_exception_preserves_identity_and_triageable_prefix(tmp_path):
    path = tmp_path / "interrupted.jsonl"
    original = RuntimeError("private producer message must not enter recording")
    with pytest.raises(RuntimeError) as caught:
        record(path, body_exception=original)
    assert caught.value is original
    report = triage_flight_recording(path, **CRITERIA)
    assert report["status"] == "unknown"
    assert report["termination"] == {"status": "producer_exception", "exception_type": "builtins.RuntimeError"}
    assert report["coverage"]["requested_run_coverage"] == "unknown"
    interruption = findings(report, "recording_interruption")
    assert len(interruption) == 1 and interruption[0]["status"] == "unknown"
    assert interruption[0]["event_ids"] == ["f1"]
    assert report["counts"]["warning"] == 0 and report["counts"]["unknown"] == 1
    assert all(finding["status"] == "pass" for finding in findings(report, "tracking"))
    assert b"private producer message" not in path.read_bytes()


def test_interruption_preserves_known_warning_and_healthy_controls(tmp_path):
    path = tmp_path / "warning-then-exception.jsonl"
    original = RuntimeError("private producer message")
    with pytest.raises(RuntimeError) as caught:
        record(path, gap=.5, body_exception=original)
    assert caught.value is original
    report = triage_flight_recording(path, **CRITERIA)
    assert report["status"] == "warning"
    assert findings(report, "camera_timing")[0]["status"] == "warning"
    assert findings(report, "camera_timing")[0]["event_ids"] == ["cam0", "cam1"]
    assert findings(report, "recording_interruption")[0]["status"] == "unknown"
    assert all(finding["status"] == "pass" for finding in findings(report, "tracking"))


def test_concise_frontdoor_retains_interruption_when_warnings_fill_preview(tmp_path):
    from policy_doctor.big_bets_frontdoor import summarize

    path = tmp_path / "many-warnings-then-exception.jsonl"
    with pytest.raises(RuntimeError):
        record(path, gap=.5, change={"values": [.8]}, body_exception=RuntimeError("private message"))
    report = triage_flight_recording(path, **CRITERIA)
    summary = summarize(report)
    assert summary["status"] == "warning" and summary["counts"]["unknown"] == 1
    assert "interrupted recording" in summary["scope"] and "UNKNOWN" in summary["scope"]
    assert summary["findings_omitted"] > 0


@pytest.mark.parametrize("failure", ["footer_before", "footer_after", "close", "both"])
def test_finalization_failures_never_mask_body_or_leave_verifiable_log(tmp_path, monkeypatch, failure):
    path = tmp_path / f"failed-{failure}.jsonl"
    original = RuntimeError("private body message")
    footer_error = OSError("private footer message")
    close_error = OSError("private close message")

    class CloseFailure:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def close(self):
            self.stream.close()
            raise close_error

    with pytest.raises(RuntimeError) as caught:
        with FlightRecorder(path, clocks=CLOCKS, provenance={"kind": "controlled"}) as recorder:
            recorder.command("c", timestamp=0, clock="host", stream="policy", **vector())
            write = recorder._write

            def faulty_write(value, *, footer=False):
                if footer and failure == "footer_before":
                    raise footer_error
                write(value, footer=footer)
                if footer and failure in ("footer_after", "both"):
                    raise footer_error

            monkeypatch.setattr(recorder, "_write", faulty_write)
            if failure in ("close", "both"):
                recorder._stream = CloseFailure(recorder._stream)
            raise original
    assert caught.value is original
    assert any("OSError" in note and "recording is unusable" in note for note in original.__notes__)
    assert all("private" not in note for note in original.__notes__)
    assert b"flight_recording_end" not in path.read_bytes()
    with pytest.raises(InputError):
        triage_flight_recording(path)


def test_normal_body_close_failure_is_raised_and_footer_invalidated(tmp_path):
    path = tmp_path / "close-failure.jsonl"
    close_error = OSError("private close message")

    class CloseFailure:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def close(self):
            self.stream.close()
            raise close_error

    with pytest.raises(OSError) as caught:
        with FlightRecorder(path, clocks=CLOCKS, provenance={"kind": "controlled"}) as recorder:
            recorder.command("c", timestamp=0, clock="host", stream="policy", **vector())
            recorder._stream = CloseFailure(recorder._stream)
    assert caught.value is close_error
    assert b"flight_recording_end" not in path.read_bytes()
    with pytest.raises(InputError):
        triage_flight_recording(path)


def test_exception_type_is_bounded_without_inspecting_message(tmp_path):
    exception_type = type("LongProducerFailure" * 30, (RuntimeError,), {
        "__str__": lambda self: (_ for _ in ()).throw(AssertionError("message inspected"))})
    original = exception_type("private message")
    path = tmp_path / "bounded-type.jsonl"
    with pytest.raises(exception_type) as caught:
        record(path, body_exception=original)
    assert caught.value is original
    report = triage_flight_recording(path, **CRITERIA)
    assert len(report["termination"]["exception_type"]) == 128
    assert b"private message" not in path.read_bytes()


def test_legacy_v1_clean_recording_keeps_original_report_semantics(tmp_path):
    current = record(tmp_path / "v2.jsonl")
    lines = current.read_bytes().splitlines(keepends=True)
    header, footer = json.loads(lines[0]), json.loads(lines[-1])
    header["schema_version"] = 1
    lines[0] = (json.dumps(header, sort_keys=True) + "\n").encode()
    del footer["termination"]
    footer["sha256"] = hashlib.sha256(b"".join(lines[:-1])).hexdigest()
    lines[-1] = (json.dumps(footer, sort_keys=True) + "\n").encode()
    legacy = tmp_path / "v1.jsonl"
    legacy.write_bytes(b"".join(lines))
    old = triage_flight_recording(legacy, **CRITERIA)
    new = triage_flight_recording(current, **CRITERIA)
    assert old["status"] == new["status"] == "pass"
    assert old["schema_version"] == 1 and "termination" not in old and "coverage" not in old
    assert old["findings"] == new["findings"] and old["counts"] == new["counts"]


@pytest.mark.parametrize("change", [{"status": "normal", "exception_type": "builtins.RuntimeError"},
                                     {"status": "producer_exception", "exception_type": "raw message!"},
                                     {"status": "unknown", "exception_type": None}])
def test_invalid_v2_termination_refused(tmp_path, change):
    path = record(tmp_path / "invalid-termination.jsonl")
    lines = path.read_bytes().splitlines(keepends=True)
    footer = json.loads(lines[-1])
    footer["termination"] = change
    lines[-1] = (json.dumps(footer) + "\n").encode()
    path.write_bytes(b"".join(lines))
    with pytest.raises(InputError, match="termination|type"):
        triage_flight_recording(path)


def test_v2_termination_status_is_bound_to_footer_hash(tmp_path):
    path = tmp_path / "tampered-termination.jsonl"
    with pytest.raises(RuntimeError):
        record(path, body_exception=RuntimeError("private message"))
    lines = path.read_bytes().splitlines(keepends=True)
    footer = json.loads(lines[-1])
    footer["termination"] = {"status": "normal", "exception_type": None}
    lines[-1] = (json.dumps(footer) + "\n").encode()
    path.write_bytes(b"".join(lines))
    with pytest.raises(InputError, match="hash"):
        triage_flight_recording(path)


def test_exclusive_creation_missing_parent_fifo_and_symlink(tmp_path):
    kwargs = {"clocks": CLOCKS, "provenance": {"kind": "controlled"}}
    regular = tmp_path / "regular"
    regular.write_text("do not overwrite")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    link = tmp_path / "link"
    link.symlink_to(regular)
    for path in (regular, fifo, link, tmp_path / "missing" / "record.jsonl"):
        with pytest.raises(InputError, match="exclusive"):
            with FlightRecorder(path, **kwargs):
                pytest.fail("must not enter")
    for path in (fifo, link, tmp_path / "absent"):
        with pytest.raises(InputError, match="regular"):
            triage_flight_recording(path)
    assert regular.read_text() == "do not overwrite"


def test_event_budget_ids_and_invalid_payloads(tmp_path):
    with FlightRecorder(tmp_path / "bounded", clocks=CLOCKS, provenance={"kind": "controlled"}, max_events=1) as recorder:
        recorder.command("c", timestamp=0, clock="host", stream="policy", **vector())
        for event_id in ("c", "other"):
            with pytest.raises(InputError, match="Duplicate|budget"):
                recorder.command(event_id, timestamp=0, clock="host", stream="policy", **vector())
        for fields in (vector(values=[float("nan")]), vector(units=[]), vector(joint_names=["elbow", "elbow"]),
                       vector(semantics="")):
            with pytest.raises(InputError):
                recorder.command("bad", timestamp=0, clock="host", stream="policy", **fields)


def test_offline_schema_and_duplicate_keys_refused_even_with_new_hash(tmp_path):
    path = record(tmp_path / "schema.jsonl")
    lines = path.read_bytes().splitlines(keepends=True)
    for bad in (b'{"kind":"interface","kind":"command"}\n', b'[]\n', b'{"kind": []}\n'):
        changed = [*lines]
        changed[1] = bad
        footer = json.loads(changed[-1])
        seal = (json.dumps({"events": footer["events"], "termination": footer["termination"]}, sort_keys=True) + "\n").encode()
        footer["sha256"] = hashlib.sha256(b"".join(changed[:-1]) + seal).hexdigest()
        changed[-1] = (json.dumps(footer) + "\n").encode()
        path.write_bytes(b"".join(changed))
        with pytest.raises(InputError):
            triage_flight_recording(path)


@pytest.mark.parametrize("criteria", [{"max_latency_s": 0}, {"max_camera_gap_s": float("inf")},
                                      {"tracking_tolerance": {"elbow": {"absolute": 1}}},
                                      {"tracking_tolerance": {"elbow": {"unit": "rad", "absolute": -1}}}])
def test_invalid_criteria_rejected(tmp_path, criteria):
    with pytest.raises(InputError):
        triage_flight_recording(record(tmp_path / "criteria.jsonl"), **criteria)
