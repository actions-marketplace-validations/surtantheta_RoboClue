"""Generated call mechanics; these are not native robot-driver executions."""
import hashlib
import json
import sys
from functools import wraps
from types import SimpleNamespace

import numpy as np
import pytest

from policy_doctor import named_capture as nc
from policy_doctor.artifacts import InputError


class Engine:
    def __init__(self):
        self._ordered_action_keys = ["a.pos", "b.pos"]
        self.vector = np.array([2., -3.], np.float32)
        self.acks = 0

    def record_dispatch(self):
        self.acks += 1


class Processor:
    def __init__(self):
        self.steps = []
        self.to_transition = self.to_output = None
        self.before_step_hooks = self.after_step_hooks = ()
        self.mutate = False

    def __call__(self, data):
        action, _ = data
        if self.mutate:
            action["a.pos"] += 10
        return action.copy()


class Backend:
    def __init__(self):
        self.mode = "echo"
        self.calls = 0

    def send_action(self, action):
        self.calls += 1
        if self.mode == "error":
            raise RuntimeError("driver sentinel")
        if self.mode == "none":
            return None
        if self.mode == "clip":
            return {k: min(1., max(-1., v)) for k, v in reversed(list(action.items()))}
        return action.copy()


class Wrapper:
    def __init__(self, inner):
        self.inner = inner

    def send_action(self, action):
        return self.inner.send_action(action)


def next_action(engine, interpolator):
    return engine.vector.copy()


def send_next_action(obs_processed, obs_raw, ctx, interpolator):
    if getattr(ctx, "denied", False):
        return None
    interp = next_action(ctx.policy.inference, interpolator)
    action_dict = {name: interp[i].item() for i, name in enumerate(ctx.data.ordered_action_keys)}
    processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
    ctx.hardware.robot_wrapper.send_action(processed)
    ctx.policy.inference.record_dispatch()
    return action_dict


@pytest.fixture
def actors(monkeypatch):
    engine, processor, backend = Engine(), Processor(), Backend()
    ctx = SimpleNamespace(policy=SimpleNamespace(inference=engine),
                          data=SimpleNamespace(ordered_action_keys=["a.pos", "b.pos"]),
                          processors=SimpleNamespace(robot_action_processor=processor),
                          hardware=SimpleNamespace(robot_wrapper=Wrapper(backend)))
    interp = SimpleNamespace(emitted_policy_action=True)
    raw = b"# Generated mechanics fixture, not native source\n"
    definition = {"functions": {"tick": send_next_action, "next": next_action,
                                "pipeline": Processor.__call__, "wrapper": Wrapper.send_action,
                                "driver": Backend.send_action, "ack": Engine.record_dispatch},
                  "sources": {"driver_source.py": raw}, "paths": {}, "consumer_argument": "action"}
    monkeypatch.setattr(nc, "_definition", lambda *args: definition)
    return ctx, interp, backend, processor, engine


def run(actors, path):
    ctx, interp, _, _, _ = actors
    with nc.capture_native_actions(ctx, interp, path, driver_namespace="generated/test") as capture:
        result = send_next_action({}, {}, ctx, interp)
    return capture, result, json.loads((path / "capture.json").read_text())


def test_actual_roles_and_return_preserved(actors, tmp_path):
    capture, result, data = run(actors, tmp_path / "capture")
    assert capture.receipt["published"] and result == {"a.pos": 2., "b.pos": -3.}
    tick = data["ticks"][0]
    assert all(tick["relations"].values())
    assert tick["driver_return"] == result and tick["ack_count"] == 1
    assert actors[-1].acks == actors[2].calls == 1
    assert data["provenance"]["recorded_action"] == "unknown"
    assert sys.getprofile() is None


def test_lawful_clipping_keeps_original_and_returned_roles(actors, tmp_path):
    actors[2].mode = "clip"
    _, result, data = run(actors, tmp_path / "capture")
    tick = data["ticks"][0]
    assert tick["driver_return"] == {"a.pos": 1., "b.pos": -1.}
    assert tick["canonical_at_processor"] == result == {"a.pos": 2., "b.pos": -3.}
    assert all(tick["relations"].values())


def test_processor_mutation_does_not_rewrite_earlier_snapshot(actors, tmp_path):
    actors[3].mutate = True
    _, result, data = run(actors, tmp_path / "capture")
    tick = data["ticks"][0]
    assert tick["canonical_at_processor"]["a.pos"] == 2.
    assert tick["canonical_at_return"]["a.pos"] == result["a.pos"] == 12.
    assert tick["relations"]["canonical_return_is_processor_input"] is True


def test_normal_none_is_success_without_fabricated_return(actors, tmp_path):
    actors[2].mode = "none"
    _, _, data = run(actors, tmp_path / "capture")
    tick = data["ticks"][0]
    assert tick["driver_completed"] and tick["wrapper_completed"]
    assert tick["driver_return"] is None and tick["outcome"] == "sent"


def test_caught_driver_exception_preserves_error_without_ack(actors, tmp_path):
    ctx, interp, backend, _, _ = actors
    backend.mode = "error"
    with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
        with pytest.raises(RuntimeError, match="driver sentinel"):
            send_next_action({}, {}, ctx, interp)
    tick = json.loads((tmp_path / "capture/capture.json").read_text())["ticks"][0]
    assert cap.receipt["published"] and tick["completion"] == "error"
    assert tick["driver_called"] and not tick["driver_completed"] and tick["ack_count"] == 0


def test_escaping_exception_no_publication(actors, tmp_path):
    ctx, interp, backend, _, _ = actors
    backend.mode = "error"
    with pytest.raises(RuntimeError, match="driver sentinel"):
        with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
            send_next_action({}, {}, ctx, interp)
    assert not cap.receipt["published"] and not (tmp_path / "capture").exists()
    assert sys.getprofile() is None


def test_denied_tick_does_not_invent_driver_call(actors, tmp_path):
    actors[0].denied = True
    _, result, data = run(actors, tmp_path / "capture")
    tick = data["ticks"][0]
    assert result is None and tick["outcome"] == "not_sent"
    assert tick["interpolated"] is None and tick["driver_request"] is None
    assert actors[2].calls == 0


@pytest.mark.parametrize("change", ["names", "engine", "processor", "steps", "driver"])
def test_changed_runtime_context_refuses_packet(actors, tmp_path, change):
    ctx, interp, backend, processor, engine = actors
    with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
        send_next_action({}, {}, ctx, interp)
        if change == "names":
            ctx.data.ordered_action_keys.reverse()
        elif change == "engine":
            engine._ordered_action_keys.reverse()
        elif change == "processor":
            ctx.processors.robot_action_processor = Processor()
        elif change == "steps":
            processor.steps.append(object())
        else:
            backend.send_action = lambda action: action
    assert not cap.receipt["published"] and not (tmp_path / "capture").exists()


def test_profiler_replacement_preserved(actors, tmp_path):
    ctx, interp, _, _, _ = actors
    def replacement(*args):
        pass
    try:
        with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
            send_next_action({}, {}, ctx, interp)
            sys.setprofile(replacement)
        assert sys.getprofile() is replacement
        assert not cap.receipt["published"]
    finally:
        sys.setprofile(None)


def test_profiler_gap_cannot_recover_false_coverage(actors, tmp_path):
    ctx, interp, _, _, _ = actors
    with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
        send_next_action({}, {}, ctx, interp)
        hook = sys.getprofile()
        sys.setprofile(None)
        sys.setprofile(hook)
    assert not cap.receipt["published"]


def test_existing_capture_and_existing_profiler_refused(actors, tmp_path):
    ctx, interp, _, _, _ = actors
    run(actors, tmp_path / "capture")
    before = (tmp_path / "capture/capture.json").read_bytes()
    with pytest.raises(InputError):
        with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test"):
            pass
    assert (tmp_path / "capture/capture.json").read_bytes() == before
    try:
        sys.setprofile(lambda *args: None)
        with pytest.raises(InputError, match="Existing profiler"):
            with nc.capture_native_actions(ctx, interp, tmp_path / "new", driver_namespace="test"):
                pass
    finally:
        sys.setprofile(None)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "2", object(), 2**53 + 1])
def test_malformed_scalar_not_published(actors, tmp_path, value):
    ctx, interp, backend, _, _ = actors
    def send_action(self, action):
        return {"a.pos": value, "b.pos": 1.}
    # Generated source-definition seam already bypasses native identity solely for this mechanics test.
    backend.send_action = send_action.__get__(backend)
    original = nc._definition(ctx, interp)
    original["functions"]["driver"] = send_action
    with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
        send_next_action({}, {}, ctx, interp)
    assert not cap.receipt["published"]


def test_decorated_driver_cannot_mutate_unobserved_bridge(actors, tmp_path):
    ctx, interp, backend, _, _ = actors
    original = Backend.send_action

    @wraps(original)
    def decorated(self, action):
        action["a.pos"] += 1
        return original(self, action)

    backend.send_action = decorated.__get__(backend)
    with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
        send_next_action({}, {}, ctx, interp)
    assert not cap.receipt["published"]
    assert "decorator changed" in cap.receipt["reason"]


def test_decorated_driver_cannot_mutate_return_after_observed_body(actors, tmp_path):
    ctx, interp, backend, _, _ = actors
    original = Backend.send_action

    @wraps(original)
    def decorated(self, action):
        returned = original(self, action)
        returned["a.pos"] = 1.
        return returned

    backend.send_action = decorated.__get__(backend)
    with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
        result = send_next_action({}, {}, ctx, interp)
    assert result == {"a.pos": 2., "b.pos": -3.}
    assert not cap.receipt["published"] and "returned values" in cap.receipt["reason"]


def test_driver_cannot_overwrite_pipeline_snapshots_with_off_path_call(actors, tmp_path):
    ctx, interp, backend, processor, _ = actors

    def driver(self, action):
        processor(({"a.pos": 90., "b.pos": 99.}, {}))
        return action

    backend.send_action = driver.__get__(backend)
    nc._definition(ctx, interp)["functions"]["driver"] = driver
    with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
        send_next_action({}, {}, ctx, interp)
    assert not cap.receipt["published"] and "off-path" in cap.receipt["reason"]


def test_observed_code_replacement_refused(actors, tmp_path):
    ctx, interp, _, _, _ = actors
    original = next_action.__code__

    def other(engine, interpolator):
        return engine.vector * 2

    try:
        with nc.capture_native_actions(ctx, interp, tmp_path / "capture", driver_namespace="test") as cap:
            send_next_action({}, {}, ctx, interp)
            next_action.__code__ = other.__code__
        assert not cap.receipt["published"] and "code changed" in cap.receipt["reason"]
    finally:
        next_action.__code__ = original


def test_cpu_tensor_snapshot_requires_consent_and_bounds(actors, tmp_path):
    torch = pytest.importorskip("torch")
    ctx, interp, _, _, _ = actors
    cap = nc.capture_native_actions(ctx, interp, tmp_path / "unused", driver_namespace="test")
    cap.names = ["a.pos", "b.pos"]
    with pytest.raises(InputError, match="explicit"):
        cap._vector(torch.tensor([1., 2.], dtype=torch.float32))
    cap.allow = True
    assert cap._vector(torch.tensor([1., 2.], dtype=torch.float32)) == [1., 2.]
    with pytest.raises(InputError, match="width"):
        cap._vector(torch.zeros(3, dtype=torch.float32))
    with pytest.raises(InputError, match="float32"):
        cap._vector(torch.zeros(2, dtype=torch.float64))


def test_source_commit_and_bounded_ticks(actors, tmp_path, monkeypatch):
    _, _, data = run(actors, tmp_path / "capture")
    raw = (tmp_path / "capture/capture.json").read_bytes()
    ready = json.loads((tmp_path / "capture/READY.json").read_text())
    assert ready["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    for entry in data["files"]:
        assert hashlib.sha256((tmp_path / "capture" / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    monkeypatch.setattr(nc, "MAX_TICKS", 1)
    ctx, interp, _, _, _ = actors
    with nc.capture_native_actions(ctx, interp, tmp_path / "over", driver_namespace="test") as cap:
        send_next_action({}, {}, ctx, interp)
        send_next_action({}, {}, ctx, interp)
    assert not cap.receipt["published"]
