"""Opt-in, owner-thread observation of the original named software action path.

No robot, processor, engine or driver calls are made by this context. Saved
source is inert evidence, never executable replay or authenticated provenance.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
import tempfile
import threading
import types
from pathlib import Path

import numpy as np

from .act_capture import _bytes, _destination, _normal_return, _sha
from .artifacts import InputError
from .command_capture import _method
from .command_capture import _native_definition as _command_definition
from .named_rules import (
    MAX_BYTES,
    MAX_DIM,
    MAX_MANIFEST,
    MAX_TICKS,
    REPOSITORY,
    REVISION,
    SOURCE_HASHES,
    SOURCE_MODULES,
)
from .publication import publish_capture

_ACTIVE = threading.local()
RELATIONS = (
    "interp_to_canonical", "canonical_to_processor_input",
    "processor_return_to_wrapper_input", "wrapper_input_to_driver_input",
    "driver_return_to_wrapper_return", "canonical_return_is_processor_input",
)


def _text(value):
    return (type(value) is str and 1 <= len(value.encode("utf8")) <= 128
            and value == value.strip() and value.isprintable())


def _names(value):
    if (type(value) is not list or not 1 <= len(value) <= MAX_DIM
            or any(not _text(v) for v in value) or len(set(value)) != len(value)):
        raise InputError("Named action order must contain distinct bounded names")
    return list(value)


def _code(function, raw):
    compiled = compile(raw, function.__code__.co_filename, "exec", dont_inherit=True)
    stack, matches = [compiled], []
    while stack:
        item = stack.pop()
        if item.co_qualname == function.__qualname__:
            matches.append(item)
        stack.extend(v for v in item.co_consts if isinstance(v, types.CodeType))
    if len(matches) != 1 or matches[0] != function.__code__:
        raise InputError("Loaded named-action code differs from pinned native source")


def _definition(ctx, interpolator):
    engine, wrapper = ctx.policy.inference, ctx.hardware.robot_wrapper
    backend, pipeline = wrapper.inner, ctx.processors.robot_action_processor
    driver = backend.send_action
    if not inspect.ismethod(driver) or driver.__self__ is not backend:
        raise InputError("Named capture requires an explicit Python driver bound method")
    definition = _command_definition(engine, interpolator, driver)
    modules = {}
    for filename in ("core.py", "robot_wrapper.py"):
        name = SOURCE_MODULES[filename]
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            raise InputError(f"Load complete native module before capture: {name}")
        path = Path(module.__file__)
        raw = _bytes(path)
        if _sha(raw) != SOURCE_HASHES[filename]:
            raise InputError(f"Unsupported native named-action source: {filename}")
        definition["sources"][filename], definition["paths"][filename] = raw, path
        modules[filename] = module
    cls = modules["robot_wrapper.py"].ThreadSafeRobot
    if type(wrapper) is not cls:
        raise InputError("Named capture requires the exact native ThreadSafeRobot")
    pipeline_cls = sys.modules["lerobot.processor.pipeline"].DataProcessorPipeline
    if type(pipeline) is not pipeline_cls:
        raise InputError("Named capture requires the actual native robot-action pipeline")
    functions = {
        "tick": modules["core.py"].send_next_action,
        "next": sys.modules["lerobot.inference.dispatch"].next_interpolated_action,
        "pipeline": _method(pipeline, "__call__", pipeline_cls),
        "wrapper": _method(wrapper, "send_action", cls),
        # Observe the actual Python body under an existing decorator, not a
        # functools.wraps name pretending to be that body's code object.
        "driver": inspect.unwrap(driver.__func__),
        "ack": definition["functions"]["ack"],
    }
    definition["loaded_bindings"] = []
    for role, function in functions.items():
        if role == "driver":
            continue
        source_name = next((n for n, mod in SOURCE_MODULES.items() if mod == function.__module__), None)
        if source_name is None:
            raise InputError("Unknown native named-action function source")
        _code(function, definition["sources"][source_name])
        definition["loaded_bindings"].append((function.__module__, sys.modules[function.__module__],
                                               function.__qualname__.split("."), function))
    driver_fn = functions["driver"]
    parameters = list(inspect.signature(driver_fn).parameters.values())
    if (len(parameters) != 2 or any(p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in parameters)
            or inspect.isgeneratorfunction(driver_fn) or inspect.iscoroutinefunction(driver_fn)
            or inspect.isasyncgenfunction(driver_fn)):
        raise InputError("Driver body must accept its owner and one explicit action argument")
    definition["consumer_argument"] = parameters[1].name
    driver_path = Path(inspect.getsourcefile(driver_fn) or "")
    driver_raw = _bytes(driver_path)
    _code(driver_fn, driver_raw)
    definition["sources"]["driver_source.py"] = driver_raw
    definition["paths"]["driver_source.py"] = driver_path
    definition["functions"] = functions
    return definition


def _pipeline_guard(pipeline):
    steps = getattr(pipeline, "steps", ())
    if type(steps) not in (list, tuple) or len(steps) > 16:
        raise InputError("Robot-action pipeline exceeds bounded step scope")
    return (tuple(steps), pipeline.to_transition, pipeline.to_output,
            tuple(getattr(pipeline, "before_step_hooks", ())),
            tuple(getattr(pipeline, "after_step_hooks", ())))


class _NamedCapture:
    def __init__(self, ctx, interpolator, output, *, driver_namespace, allow_tensor_snapshot,
                 observe_recording=False):
        if (not _text(driver_namespace) or type(allow_tensor_snapshot) is not bool
                or type(observe_recording) is not bool):
            raise InputError("Explicit bounded driver_namespace and Boolean snapshot consent required")
        self.ctx, self.interpolator, self.output = ctx, interpolator, output
        self.namespace, self.allow = driver_namespace, allow_tensor_snapshot
        self.observe_recording, self._recording = observe_recording, None
        self._used = self._uninstalling = False
        self._owner = self._hook = self._definition = None
        self._frames, self._refs, self._ticks = {}, {}, []
        self._tick = self._error = None
        self.receipt = self._unpublished("No native named tick observed")

    @staticmethod
    def _unpublished(reason):
        return {"kind": "named_action_capture_receipt", "status": "unknown",
                "published": False, "reason": reason}

    def __enter__(self):
        if self._used or getattr(_ACTIVE, "active", False) or sys.getprofile() is not None:
            raise InputError("Existing profiler, nested or reused named capture is unsupported")
        self.output = _destination(self.output)
        self.names = _names(self.ctx.data.ordered_action_keys)
        self.engine = self.ctx.policy.inference
        self.engine_names = _names(self.engine._ordered_action_keys)
        self.wrapper = self.ctx.hardware.robot_wrapper
        self.backend = self.wrapper.inner
        self.pipeline = self.ctx.processors.robot_action_processor
        self._definition = _definition(self.ctx, self.interpolator)
        if self.observe_recording:
            from .writer_capture import RecordingObservation

            self._recording = RecordingObservation(self)
        self._observed_codes = {role: fn.__code__ for role, fn in self._definition["functions"].items()}
        self._driver_inner = self._definition["functions"]["driver"]
        self._guard = _pipeline_guard(self.pipeline)
        self._actors = (self.ctx.policy, self.ctx.hardware, self.ctx.data, self.ctx.processors)
        self._native_methods = tuple((obj, name, getattr(obj, name)) for obj, name in (
            (self.wrapper, "send_action"), (self.backend, "send_action"),
            (self.pipeline, "__call__"), (self.engine, "record_dispatch")))
        if self._recording is not None:
            self._native_methods += tuple(self._recording.methods)
        self._owner, self._hook = threading.get_ident(), self._profile
        self._used = True
        _ACTIVE.active = True
        sys.setprofile(self._hook)
        return self

    def _check(self):
        if self._recording is not None:
            self._recording.check()
        for role, function in self._definition["functions"].items():
            if function.__code__ is not self._observed_codes[role]:
                raise InputError("Observed function code changed during capture")
        for module_name, module, attributes, function in self._definition.get("loaded_bindings", ()):
            if sys.modules.get(module_name) is not module:
                raise InputError("Observed native module was replaced")
            value = module
            for name in attributes:
                value = getattr(value, name, None)
            if value is not function:
                raise InputError("Observed native module/class function was replaced")
        if (self.ctx.policy is not self._actors[0] or self.ctx.hardware is not self._actors[1]
                or self.ctx.data is not self._actors[2] or self.ctx.processors is not self._actors[3]
                or self.ctx.policy.inference is not self.engine
                or self.ctx.hardware.robot_wrapper is not self.wrapper
                or self.wrapper.inner is not self.backend
                or self.ctx.processors.robot_action_processor is not self.pipeline
                or self.ctx.data.ordered_action_keys != self.names
                or self.engine._ordered_action_keys != self.engine_names):
            raise InputError("Native named-action context, actor or declared order changed")
        current = _pipeline_guard(self.pipeline)
        if (len(current[0]) != len(self._guard[0])
                or any(a is not b for a, b in zip(current[0], self._guard[0], strict=True))
                or current[1] is not self._guard[1] or current[2] is not self._guard[2]
                or current[3:] != self._guard[3:]):
            raise InputError("Robot-action processor steps, converters or hooks changed")
        for obj, name, before in self._native_methods:
            now = getattr(obj, name)
            if inspect.ismethod(before):
                if (not inspect.ismethod(now) or now.__self__ is not before.__self__
                        or now.__func__ is not before.__func__):
                    raise InputError("Observed native/driver method changed")
            elif now is not before:
                raise InputError("Observed native/driver function changed")
        current_driver = self.backend.send_action
        if inspect.ismethod(current_driver) and inspect.unwrap(current_driver.__func__) is not self._driver_inner:
            raise InputError("Driver decorator target changed during observation")

    @staticmethod
    def _mapping(value):
        if value is None:
            return None
        if type(value) is not dict or not 1 <= len(value) <= MAX_DIM:
            raise InputError("Observed named actions must be bounded plain dictionaries")
        copied = {}
        for name, number in value.items():
            if (not _text(name) or type(number) not in (int, float)
                    or (type(number) is int and abs(number) > 2**53) or not math.isfinite(number)):
                raise InputError("Named actions require finite ordinary scalar values and bounded names")
            copied[name] = float(number)
        return copied

    def _vector(self, value):
        if value is None:
            return None
        if type(value) is np.ndarray:
            if value.dtype != np.dtype("float32"):
                raise InputError("Interpolation snapshots require float32")
        else:
            torch = sys.modules.get("torch")
            if not self.allow:
                raise InputError("Tensor snapshots require explicit allow_tensor_snapshot=True")
            if (torch is None or type(value) is not torch.Tensor or value.device.type != "cpu"
                    or value.layout != torch.strided or value.dtype != torch.float32):
                raise InputError("Interpolation snapshots require ordinary CPU float32 tensors")
        if tuple(value.shape) != (len(self.names),):
            raise InputError("Interpolated vector must match the actual canonical mapping width")
        copied = value.copy() if type(value) is np.ndarray else value.detach().numpy().copy()
        if not np.isfinite(copied).all():
            raise InputError("Interpolation values must be finite")
        return copied.tolist()

    def _inside(self, frame, role):
        target, parent = self._frames.get(role), frame.f_back
        for _ in range(40):
            if parent is None:
                return False
            if parent is target:
                return True
            parent = parent.f_back
        return False

    def _start(self, frame):
        self._check()
        if self._tick is not None or len(self._ticks) >= MAX_TICKS:
            raise InputError("Nested or over-budget native named ticks")
        self._frames["tick"] = frame
        self._tick = dict.fromkeys(("interpolated", "canonical_at_processor", "canonical_at_return",
                                   "robot_processor_output", "wrapper_request", "driver_request", "driver_return",
                                   "wrapper_return"))
        self._tick.update(tick_id=len(self._ticks), completion=None, outcome=None,
                          driver_called=False, driver_completed=False, wrapper_completed=False,
                          ack_count=0, emitted_policy_action=False,
                          relations=dict.fromkeys(RELATIONS))

    def _entry(self, role, frame):
        if role in self._frames:
            raise InputError("Recursive named-action boundary")
        self._frames[role] = frame
        tick, relations = self._tick, self._tick["relations"]
        if role == "next":
            if tick["interpolated"] is not None or frame.f_back is not self._frames["tick"]:
                raise InputError("Multiple or off-path interpolation pulls in one native tick are unsupported")
        elif role == "pipeline":
            if tick["canonical_at_processor"] is not None or frame.f_back is not self._frames["tick"]:
                raise InputError("Multiple or off-path robot processor calls in one native tick are unsupported")
            data = frame.f_locals["data"]
            if type(data) is not tuple or len(data) != 2:
                raise InputError("Robot-action processor requires actual native action/observation tuple")
            value = data[0]
            tick["canonical_at_processor"] = self._mapping(value)
            self._refs["canonical"] = value
            caller = self._frames["tick"].f_locals
            relations["canonical_to_processor_input"] = caller.get("action_dict") is value
            vector = tick["interpolated"]
            relations["interp_to_canonical"] = (
                vector is not None and caller.get("interp") is self._refs.get("interp")
                and tick["canonical_at_processor"] == dict(zip(self.names, vector, strict=True)))
        elif role == "wrapper":
            value = frame.f_locals["action"]
            tick["wrapper_request"] = self._mapping(value)
            self._refs["wrapper_request"] = value
            relations["processor_return_to_wrapper_input"] = (
                value is self._refs.get("processed") and value is self._frames["tick"].f_locals.get("processed"))
        elif role == "driver":
            if tick["driver_called"]:
                raise InputError("Multiple driver calls in a canonical tick are unsupported")
            value = frame.f_locals[self._definition["consumer_argument"]]
            tick["driver_called"] = True
            tick["driver_request"] = self._mapping(value)
            if tick["driver_request"] != tick["wrapper_request"]:
                raise InputError("Unmodeled driver decorator changed the named request before its observed body")
            relations["wrapper_input_to_driver_input"] = value is self._refs.get("wrapper_request")

    def _return(self, role, frame, value):
        tick, normal = self._tick, _normal_return(frame)
        if role == "tick":
            self._check()
            tick["completion"] = "normal" if normal else "error"
            tick["outcome"] = "error" if not normal else "sent" if tick["ack_count"] == 1 else "not_sent"
            if normal:
                tick["canonical_at_return"] = self._mapping(value)
                if value is not None:
                    tick["relations"]["canonical_return_is_processor_input"] = value is self._refs.get("canonical")
            emitted = self.interpolator.emitted_policy_action
            if type(emitted) is not bool:
                raise InputError("Invalid native emitted-action marker")
            tick["emitted_policy_action"] = emitted
            if self._recording is not None:
                self._recording.tick_return(tick, value, frame, normal)
            self._ticks.append(tick)
            self._tick = None
            self._frames.clear()
            self._refs.clear()
            return
        if normal:
            if role == "next":
                tick["interpolated"] = self._vector(value)
                self._refs["interp"] = value
            elif role == "pipeline":
                tick["robot_processor_output"] = self._mapping(value)
                self._refs["processed"] = value
            elif role == "driver":
                tick["driver_return"] = self._mapping(value)
                tick["driver_completed"] = True
                self._refs["driver_return"] = value
            elif role == "wrapper":
                tick["wrapper_return"] = self._mapping(value)
                if tick["wrapper_return"] != tick["driver_return"]:
                    raise InputError("Unmodeled driver decorator changed the returned values before wrapper completion")
                tick["wrapper_completed"] = True
                tick["relations"]["driver_return_to_wrapper_return"] = (
                    tick["driver_completed"] and value is self._refs.get("driver_return"))
            elif role == "ack":
                tick["ack_count"] += 1
        self._frames.pop(role, None)

    def _profile(self, frame, event, value):
        if event == "c_call" and value is sys.setprofile and not self._uninstalling:
            self._error = "Profiler coverage gap observed"
        if self._error or event not in ("call", "return"):
            return
        try:
            if self._recording is not None and self._recording.profile(frame, event, value):
                return
            role = next((k for k, code in self._observed_codes.items() if frame.f_code is code), None)
            if role is None:
                return
            if role == "tick":
                if frame.f_locals.get("ctx") is not self.ctx or frame.f_locals.get("interpolator") is not self.interpolator:
                    return
                if event == "call":
                    self._start(frame)
                elif self._frames.get("tick") is frame:
                    self._return(role, frame, value)
                return
            if self._tick is None or not self._inside(frame, "tick"):
                return
            owners = {"pipeline": self.pipeline, "wrapper": self.wrapper, "driver": self.backend,
                      "ack": self.engine}
            if role in owners and frame.f_locals.get("self") is not owners[role]:
                return
            if role == "next" and (frame.f_locals.get("engine") is not self.engine
                                   or frame.f_locals.get("interpolator") is not self.interpolator):
                return
            if role == "driver" and not self._inside(frame, "wrapper"):
                raise InputError("Driver call outside the observed wrapper send boundary, including hold, is unsupported")
            if event == "call":
                self._entry(role, frame)
            elif self._frames.get(role) is frame:
                self._return(role, frame, value)
        except Exception as exc:
            self._error = f"Named capture failed: {type(exc).__name__}: {str(exc)[:512]}"
            self._frames.clear()
            self._refs.clear()

    def _write(self):
        self._check()
        for name, path in self._definition["paths"].items():
            if _bytes(path) != self._definition["sources"][name]:
                raise InputError("Native or driver source changed during observation")
        output = _destination(self.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".named-action-", dir=output.parent) as directory:
            stage = Path(directory)
            (stage / "sources").mkdir()
            for name, raw in self._definition["sources"].items():
                if not name or Path(name).name != name:
                    raise InputError("Unsafe source filename")
                (stage / "sources" / name).write_bytes(raw)
            recording = self._recording.snapshot(stage) if self._recording is not None else None
            sources = {name: {"sha256": _sha(raw)} for name, raw in self._definition["sources"].items()}
            files = [{"path": f"sources/{name}", **entry} for name, entry in sorted(sources.items())]
            if recording is not None:
                files += recording['dataset_files']
            packet = {"schema_version": 2 if recording is not None else 1,
                      "kind": "named_action_capture", "domain": "native_named_action_software",
                      "source": {"repository": REPOSITORY, "revision": REVISION, "files": sources},
                      "action_names": self.names, "engine_action_names": self.engine_names,
                      "driver_namespace": self.namespace,
                      "coverage": {"target": "one_native_send_next_action_on_owner_thread",
                                   "complete_target_call_history": True, "external_state_mutations": "untracked"},
                      "ticks": self._ticks, "files": files,
                      "provenance": {"origin": "caller-runtime-evidence-not-attestation",
                                     "selected": "unknown", "policy_processor": "unknown",
                                     "physical_execution": "unknown", "recorded_action": "unknown"}}
            if recording is not None:
                packet['recording'] = recording
            raw = json.dumps(packet, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
            if (len(raw) > MAX_MANIFEST
                    or len(raw) + sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()) > MAX_BYTES):
                raise InputError("Named capture exceeds bounded manifest or total source bytes")
            (stage / "capture.json").write_bytes(raw)
            (stage / "READY.json").write_text(json.dumps({"schema_version": 1, "manifest_sha256": _sha(raw)}))
            publish_capture(stage, output)
        return {"kind": "named_action_capture_receipt", "status": "captured", "published": True,
                "directory": str(output), "tick_count": len(self._ticks), "domain": "native_named_action_software"}

    def __exit__(self, exc_type, exc, tb):
        if threading.get_ident() != self._owner:
            raise InputError("Named capture must exit on the owner thread")
        owned = sys.getprofile() is self._hook
        if owned:
            self._uninstalling = True
            sys.setprofile(None)
        _ACTIVE.active = False
        try:
            if exc_type is not None:
                self.receipt = self._unpublished("Caller exception preserved; no capture published")
            elif not owned:
                self.receipt = self._unpublished("Profiler replaced; replacement preserved")
            elif self._error or self._tick is not None or not self._ticks:
                self.receipt = self._unpublished(self._error or "No complete native named tick captured")
            else:
                self.receipt = self._write()
        except Exception as failure:
            self.receipt = self._unpublished(f"Publication failed: {type(failure).__name__}: {str(failure)[:512]}")
        finally:
            if self._recording is not None:
                self._recording.clear()
                self._recording = None
            self._frames.clear()
            self._refs.clear()
            self._ticks.clear()
            self.ctx = self.interpolator = self.engine = self.wrapper = self.backend = self.pipeline = None
            self._definition = self._tick = self._hook = self._native_methods = self._guard = self._actors = self._driver_inner = None
            self._observed_codes = None
        return False


def capture_native_actions(ctx, interpolator, output, *, driver_namespace, allow_tensor_snapshot=False,
                           observe_recording=False):
    """Observe existing native send_next_action calls; never call a robot or engine.

    Only already loaded pinned synchronous ACT, native robot processor/wrapper
    and an explicit Python driver method are supported. Inspect ``receipt`` on
    the returned context after exit. Writer/physical roles remain unknown.
    """
    return _NamedCapture(ctx, interpolator, output, driver_namespace=driver_namespace,
                         allow_tensor_snapshot=allow_tensor_snapshot, observe_recording=observe_recording)
