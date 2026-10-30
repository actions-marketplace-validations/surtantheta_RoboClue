"""Offline observed native selection, processor and software-consumer chains.

Saved call witnesses establish scoped local associations, not runtime authenticity,
physical commands, named-joint mappings or the absence of external writes.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy

import numpy as np

from .artifacts import InputError
from .command_rules import (
    MAX_BYTES,
    MAX_DIM,
    MAX_MANIFEST,
    MAX_MULTIPLIER,
    MAX_TICKS,
    REPOSITORY,
    REVISION,
    SOURCE_HASHES,
)
from .layouts import _read, _safe_root, _tolerances
from .selection import _arrays, _digest, _parse

SCOPE = (
    "Observed native synchronous selection→effective MEAN_STD processor→interpolated software consumer. "
    "Ordered names describe the declared engine tensor order, not an executed named-joint mapping. "
    "Captured-call ordinal is not episode/time alignment. Source hashes and call witnesses do not authenticate "
    "runtime origin or exclude external writes. Historical query lineage, recorded actions, measured motion "
    "and task competence remain unknown. A changed legal recipe or consumer return is not intrinsically a fault."
)
ROLES = (
    "selected",
    "post_input",
    "post_output",
    "engine_output",
    "add_input",
    "get_output",
    "requested",
    "returned",
    "dispatch_return",
)
RELATIONS = (
    "selected_to_post_input",
    "post_return_is_engine_action",
    "engine_to_add_input",
    "get_prebuffer_to_return",
    "get_return_to_consumer",
    "consumer_to_dispatch_target",
)


def _text(value):
    return (
        type(value) is str
        and 1 <= len(value.encode("utf8")) <= 128
        and value.isprintable()
        and value == value.strip()
    )


def _names(value):
    if (
        type(value) is not list
        or not 1 <= len(value) <= MAX_DIM
        or any(not _text(v) for v in value)
        or len(set(value)) != len(value)
    ):
        raise InputError("Command action names require unique bounded declared tensor coordinates")
    return list(value)


def _token(value):
    if value is not None and (
        type(value) is not str or not re.fullmatch(r"o[0-9]{6}", value) or int(value[1:]) >= 4096
    ):
        raise InputError("Invalid opaque command token")
    return value


def _load(capture):
    root = _safe_root(capture)
    expected_names = {"capture.json", "READY.json", "arrays.npz"} | {"sources/" + n for n in SOURCE_HASHES}
    tree = set()
    for path in root.iterdir():
        if path.is_symlink():
            raise InputError("Command capture symlinks refused")
        if path.name == "sources" and path.is_dir():
            for source in path.iterdir():
                if source.is_symlink() or not source.is_file() or source.name not in SOURCE_HASHES:
                    raise InputError("Unexpected command source payload")
                tree.add("sources/" + source.name)
        elif path.is_file() and path.name in expected_names:
            tree.add(path.name)
        else:
            raise InputError("Unexpected command capture entry")
    if tree != expected_names:
        raise InputError("Command capture requires its complete bounded file inventory")
    raw = _read(root / "capture.json", MAX_MANIFEST)
    ready_raw = _read(root / "READY.json", 4096)
    data, ready = _parse(raw), _parse(ready_raw)
    if (
        set(ready) != {"schema_version", "manifest_sha256"}
        or type(ready["schema_version"]) is not int
        or ready["schema_version"] != 1
        or ready["manifest_sha256"] != _digest(raw)
    ):
        raise InputError("Command commit marker/hash mismatch")
    if (
        data.get("kind") != "command_path_capture"
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
        or data.get("domain") != "software_consumer"
        or type(data.get("ticks")) is not list
        or not 1 <= len(data["ticks"]) <= MAX_TICKS
    ):
        raise InputError("Unsupported command-path capture schema/domain/tick count")
    names = _names(data.get("action_names"))
    if not _text(data.get("consumer_namespace")):
        raise InputError("Command consumer namespace must be explicit bounded text")
    entries = data.get("files")
    if type(entries) is not list or len(entries) != len(SOURCE_HASHES) + 1:
        raise InputError("Command payload inventory is incomplete")
    hashes = {"capture.json": _digest(raw), "READY.json": _digest(ready_raw)}
    payloads, total = {}, len(raw) + len(ready_raw)
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"path", "sha256"}:
            raise InputError("Malformed command payload entry")
        name = entry["path"]
        if (
            type(name) is not str
            or name not in expected_names - {"capture.json", "READY.json"}
            or name in payloads
            or type(entry["sha256"]) is not str
            or not re.fullmatch("[0-9a-f]{64}", entry["sha256"])
        ):
            raise InputError("Invalid command payload path/hash")
        value = _read(root / name, MAX_BYTES if name == "arrays.npz" else 512 * 1024)
        total += len(value)
        if total > MAX_BYTES or _digest(value) != entry["sha256"]:
            raise InputError("Command payload hash mismatch or total byte budget exceeded")
        payloads[name], hashes[name] = value, _digest(value)
    source = data.get("source")
    if type(source) is not dict or source.get("files") != {
        n: {"sha256": hashes["sources/" + n]} for n in SOURCE_HASHES
    }:
        raise InputError("Command source identity and payload hashes differ")
    supported = (
        source.get("repository") == REPOSITORY
        and source.get("revision") == REVISION
        and all(hashes["sources/" + n] == h for n, h in SOURCE_HASHES.items())
    )
    arrays = _arrays(payloads["arrays.npz"], data.get("arrays"))
    for value in arrays.values():
        if value.dtype != np.dtype("float32") or value.shape not in ((len(names),), (1, len(names))):
            raise InputError("Command arrays require finite float32 batch-one vectors in declared order")
    return root, data, arrays, hashes, supported


def _recheck(root, hashes):
    for name, digest in hashes.items():
        if _digest(_read(root / name, MAX_BYTES)) != digest:
            raise InputError("Command evidence changed during inspection")
    # Re-enumerate too: an inserted file or symlink is not part of the frozen snapshot.
    current = set()
    for p in root.iterdir():
        if p.is_symlink():
            raise InputError("Command evidence changed to a symlink")
        if p.name == "sources" and p.is_dir():
            for q in p.iterdir():
                if q.is_symlink() or not q.is_file():
                    raise InputError("Command source tree changed")
                current.add("sources/" + q.name)
        elif p.is_file():
            current.add(p.name)
        else:
            raise InputError("Command evidence tree changed")
    if current != set(hashes):
        raise InputError("Command evidence inventory changed")


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and abs(value) <= np.finfo(np.float32).max
    except (TypeError, ValueError, OverflowError):
        return False


def _processor_contract(value):
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"mode", "mean", "std"} or value["mode"] != "MEAN_STD":
        raise InputError("Expected processor requires explicit MEAN_STD mean/std vectors")
    result = {"mode": "MEAN_STD"}
    for name in ("mean", "std"):
        v = value[name]
        if type(v) is not list or not 1 <= len(v) <= MAX_DIM or any(not _finite_number(n) for n in v):
            raise InputError(
                "Expected processor statistics require bounded finite numeric vectors, excluding bool"
            )
        result[name] = list(v)
    if len(result["mean"]) != len(result["std"]):
        raise InputError("Expected mean/std widths differ")
    return result


def _state(state, a):
    if (
        type(state) is not dict
        or set(state) != {"previous", "previous_token", "buffer", "buffer_tokens", "index", "emitted"}
        or type(state["buffer"]) is not list
        or len(state["buffer"]) > MAX_MULTIPLIER
        or type(state["buffer_tokens"]) is not list
        or len(state["buffer_tokens"]) != len(state["buffer"])
        or type(state["index"]) is not int
        or not 0 <= state["index"] <= len(state["buffer"])
        or type(state["emitted"]) is not bool
    ):
        raise InputError("Malformed native interpolation state")
    if (state["previous"] is None) != (state["previous_token"] is None):
        raise InputError("Interpolation previous object/ref disagree")
    if state["previous"] is not None:
        a(state["previous"], 1)
    _token(state["previous_token"])
    for ref, token in zip(state["buffer"], state["buffer_tokens"], strict=True):
        a(ref, 1)
        _token(token)
    return state


def _view_binding(view, tokens, a, post_ref, engine_ref):
    if view is None:
        return False
    if type(view) is not dict or set(view) != {"post", "engine", "same_live_storage", "basis"}:
        raise InputError("Malformed postprocessor/engine view witness")
    if type(view["same_live_storage"]) is not bool:
        raise InputError("Live-storage relation requires a strict Boolean")
    metadata = []
    for key, ref, rank in [("post", post_ref, 2), ("engine", engine_ref, 1)]:
        item = view[key]
        if (
            type(item) is not dict
            or set(item) != {"base_token", "shape", "stride", "storage_offset", "element_size"}
            or type(item["shape"]) is not list
            or any(type(v) is not int for v in item["shape"])
            or item["shape"] != list(a(ref, rank).shape)
            or type(item["stride"]) is not list
            or len(item["stride"]) != rank
            or any(type(v) is not int or not 0 <= v <= 2**53 - 1 for v in item["stride"])
            or type(item["storage_offset"]) is not int
            or not 0 <= item["storage_offset"] <= 2**53 - 1
            or type(item["element_size"]) is not int
            or item["element_size"] != 4
        ):
            raise InputError("Invalid squeeze view layout metadata")
        _token(item["base_token"])
        metadata.append(item)
    post, engine = metadata
    return (
        view["same_live_storage"] is True
        and view["basis"] == "live_post_output_storage"
        and post["base_token"] is not None
        and post["base_token"] == engine["base_token"] == tokens["post_output"]
        and engine["stride"] == post["stride"][1:]
        and post["storage_offset"] == engine["storage_offset"]
        and (a(engine_ref, 1).size == 1 or engine["stride"][0] > 0)
    )


def _analyze(data, arrays, supported, atol, rtol):
    coverage = data.get("coverage")
    covered = (
        type(coverage) is dict
        and coverage.get("target") == "one_sync_engine_dispatch_on_owner_thread"
        and coverage.get("complete_target_call_history") is True
        and coverage.get("external_state_mutations") == "untracked"
    )
    events, buffer_origins, previous_origins = [], {}, {}

    def a(ref, rank=None):
        if type(ref) is not str or ref not in arrays or (rank is not None and arrays[ref].ndim != rank):
            raise InputError("Missing or wrongly shaped command array reference")
        return arrays[ref]

    def equal(x, y):
        return x.shape == y.shape and np.allclose(x, y, atol=atol, rtol=rtol)

    for ordinal, tick in enumerate(data["ticks"]):
        if (
            type(tick) is not dict
            or type(tick.get("tick_id")) is not int
            or tick["tick_id"] != ordinal
            or tick.get("completion") not in ("normal", "error")
            or tick.get("outcome") not in ("sent", "not_sent", "error")
            or type(tick.get("consumer_called")) is not bool
            or type(tick.get("consumer_completed")) is not bool
            or type(tick.get("ack_count")) is not int
            or tick["ack_count"] not in (0, 1)
        ):
            raise InputError("Malformed command tick identity/completion/acknowledgement")
        for key, rank in [
            ("selected", 2),
            ("engine_output", 1),
            ("requested", 1),
            ("returned", 1),
            ("dispatch_return", 1),
        ]:
            if tick.get(key) is not None:
                a(tick[key], rank)
        if tick["consumer_completed"] and not tick["consumer_called"]:
            raise InputError("A completed consumer must actually have been called")
        if tick["outcome"] == "sent" and (
            tick["completion"] != "normal" or not tick["consumer_completed"] or tick["ack_count"] != 1
        ):
            raise InputError("A sent tick requires a completed consumer and native acknowledgement")
        if tick["outcome"] != "sent" and tick["ack_count"] != 0:
            raise InputError("Unsent/error tick cannot claim native acknowledgement")
        numeric, reasons = [], []
        bindings = tick.get("bindings")
        if (
            type(bindings) is not dict
            or type(bindings.get("tokens")) is not dict
            or set(bindings["tokens"]) != set(ROLES)
            or type(bindings.get("relations")) is not dict
            or set(bindings["relations"]) != set(RELATIONS)
        ):
            raise InputError("Command tick requires bounded opaque role and relation witnesses")
        tokens, relations = bindings["tokens"], bindings["relations"]
        for value in tokens.values():
            _token(value)
        if any(v is not None and type(v) is not bool for v in relations.values()):
            raise InputError("Command binding relations require nullable strict Booleans")
        pp = tick.get("postprocessor")
        observed_processor, processor_supported = None, False
        pp_input = pp_output = None
        if pp is not None:
            if (
                type(pp) is not dict
                or type(pp.get("steps")) is not list
                or len(pp["steps"]) > 16
                or type(pp.get("hooks_present")) is not bool
                or type(pp.get("converters_supported")) is not bool
            ):
                raise InputError("Malformed actual postprocessor snapshot")
            pp_input, pp_output = pp.get("input"), pp.get("output")
            a(pp_input, 2)
            a(pp_output, 2)
            steps = pp["steps"]
            processor_supported = (
                not pp["hooks_present"]
                and pp["converters_supported"]
                and len(steps) == 2
                and [s.get("kind") for s in steps if type(s) is dict] == ["unnormalizer", "device"]
            )
            processor_numeric = []
            prior = pp_input
            for index, step in enumerate(steps):
                if type(step) is not dict or type(step.get("index")) is not int or step["index"] != index:
                    raise InputError("Processor steps must retain native call order")
                si, so = step.get("input"), step.get("output")
                a(si, 2)
                a(so, 2)
                processor_numeric.append(equal(a(prior), a(si)))
                prior = so
                cfg = step.get("configuration")
                if type(cfg) is not dict:
                    raise InputError("Processor step configuration is missing")
                if step["kind"] == "unnormalizer":
                    transform = step.get("transform")
                    qualified_transform = (
                        type(transform) is dict
                        and transform.get("inverse") is True
                        and transform.get("key") == "action"
                        and cfg.get("mode") == "MEAN_STD"
                        and cfg.get("device") == "cpu"
                    )
                    if qualified_transform:
                        stats = transform.get("stats")
                        if type(stats) is not dict:
                            raise InputError("Actual transform statistics must be an array-reference mapping")
                        if set(stats) != {"mean", "std"}:
                            processor_supported = False
                            continue
                        mean, std = a(stats["mean"], 1), a(stats["std"], 1)
                        ti, to = a(transform.get("input"), 2), a(transform.get("output"), 2)
                        with np.errstate(over="ignore", invalid="ignore"):
                            expected = ti * std + mean
                        processor_numeric.extend([equal(a(si), ti), equal(expected, to), equal(to, a(so))])
                        observed_processor = {"mode": "MEAN_STD", "mean": mean.tolist(), "std": std.tolist()}
                    else:
                        processor_supported = False
                elif step["kind"] == "device":
                    processor_supported = (
                        processor_supported
                        and cfg.get("device") == "cpu"
                        and cfg.get("float_dtype") in (None, "float32", "torch.float32")
                    )
                    processor_numeric.append(equal(a(si), a(so)))
                else:
                    processor_supported = False
            processor_numeric.append(equal(a(prior), a(pp_output)))
            if tick.get("selected") is not None:
                processor_numeric.append(equal(a(tick["selected"]), a(pp_input)))
            if tick.get("engine_output") is not None:
                processor_numeric.append(equal(a(pp_output)[0], a(tick["engine_output"])))
            # Unknown hooks/converters/recipes may legally change the action between
            # these native calls. Do not manufacture a numeric contradiction there.
            if processor_supported:
                numeric.extend(processor_numeric)
        interp = tick.get("interpolation")
        if (
            type(interp) is not dict
            or type(interp.get("multiplier")) is not int
            or not 1 <= interp["multiplier"] <= MAX_MULTIPLIER
        ):
            raise InputError("Command interpolation requires a bounded strict multiplier")
        pre = _state(interp.get("pre"), a)
        add, get = interp.get("add"), interp.get("get")
        get_binding, add_binding, origin = False, False, None
        requires_previous, previous_origin = False, None
        if add is not None:
            if type(add) is not dict:
                raise InputError("Malformed interpolation add observation")
            ai, post = a(add.get("input"), 1), _state(add.get("post"), a)
            old = None if pre["previous"] is None else a(pre["previous"])
            m = interp["multiplier"]
            with np.errstate(over="ignore", invalid="ignore"):
                expected_buffer = (
                    [old + np.float32(i / m) * (ai - old) for i in range(1, m)]
                    if m > 1 and old is not None
                    else []
                ) + [ai]
            numeric.extend(
                [
                    len(post["buffer"]) == len(expected_buffer),
                    post["index"] == 0,
                    post["emitted"] == pre["emitted"],
                    post["previous"] is not None and equal(a(post["previous"]), ai),
                ]
            )
            numeric.extend(
                equal(a(ref), value) for ref, value in zip(post["buffer"], expected_buffer, strict=False)
            )
            if tick.get("engine_output") is not None:
                numeric.append(equal(a(tick["engine_output"]), ai))
            add_binding = (
                relations["engine_to_add_input"] is True
                and tokens["engine_output"] is not None
                and tokens["engine_output"] == tokens["add_input"]
            )
            # Match both cloned-buffer values and object identity to their actual
            # native add. An in-place mutation retains its token but loses this
            # original transform association, so it must stay unknown.
            prior = previous_origins.get(pre["previous_token"])
            prior_origin = (
                prior[0]
                if prior is not None
                and pre["previous"] is not None
                and np.array_equal(a(pre["previous"]), a(prior[1]))
                else None
            )
            for index, (token, ref) in enumerate(zip(post["buffer_tokens"], post["buffer"], strict=True)):
                if token is not None:
                    buffer_origins[token] = {
                        "origin": ordinal,
                        "ref": ref,
                        "requires_previous": m > 1 and old is not None and index < len(post["buffer"]) - 1,
                        "previous_origin": prior_origin,
                    }
            if post["previous_token"] is not None:
                previous_origins[post["previous_token"]] = (ordinal, post["previous"])
            state_before_get = post
        else:
            state_before_get = pre
        emitted = False
        if get is not None:
            if type(get) is not dict:
                raise InputError("Malformed interpolation get observation")
            gp, go = _state(get.get("pre"), a), _state(get.get("post"), a)
            numeric.append(_states_equal(state_before_get, gp, a, equal))
            index = gp["index"]
            if index < len(gp["buffer"]):
                out = get.get("output")
                a(out, 1)
                emitted = index + 1 == len(gp["buffer"])
                numeric.extend(
                    [
                        equal(a(out), a(gp["buffer"][index])),
                        go["index"] == index + 1,
                        go["emitted"] == emitted,
                        _states_values_equal(gp, go, a, equal),
                    ]
                )
                buffer_token = gp["buffer_tokens"][index]
                get_binding = (
                    relations["get_prebuffer_to_return"] is True
                    and buffer_token is not None
                    and buffer_token == tokens["get_output"]
                )
                witness = buffer_origins.get(buffer_token) if get_binding else None
                if witness is not None and np.array_equal(a(gp["buffer"][index]), a(witness["ref"])):
                    origin = witness["origin"]
                    requires_previous = witness["requires_previous"]
                    previous_origin = witness["previous_origin"]
            else:
                if get.get("output") is not None:
                    raise InputError("Exhausted interpolation buffer cannot return an array")
                numeric.extend(
                    [go["index"] == index, go["emitted"] is False, _states_values_equal(gp, go, a, equal)]
                )
        if tick.get("requested") is not None and get is not None and get.get("output") is not None:
            numeric.append(equal(a(tick["requested"]), a(get["output"])))
        if tick.get("dispatch_return") is not None and tick.get("requested") is not None:
            numeric.append(equal(a(tick["dispatch_return"]), a(tick["requested"])))
        role_refs = {
            "selected": tick.get("selected"),
            "post_input": pp_input,
            "post_output": pp_output,
            "engine_output": tick.get("engine_output"),
            "add_input": None if add is None else add.get("input"),
            "get_output": None if get is None else get.get("output"),
            "requested": tick.get("requested"),
            "returned": tick.get("returned"),
            "dispatch_return": tick.get("dispatch_return"),
        }
        # The same witnessed live object cannot simultaneously have contradictory role values/shapes.
        seen = {}
        for role, ref in role_refs.items():
            token = tokens[role]
            if token is not None and ref is not None:
                if token in seen and not np.array_equal(a(ref), a(seen[token])):
                    raise InputError("One opaque object token has incompatible command role arrays")
                seen[token] = ref
        local = (
            tick.get("selected") is not None
            and pp is not None
            and tick.get("engine_output") is not None
            and relations["selected_to_post_input"] is True
            and tokens["selected"] is not None
            and tokens["selected"] == tokens["post_input"]
            and relations["post_return_is_engine_action"] is True
            and _view_binding(
                bindings.get("post_to_engine_view"), tokens, a, pp_output, tick["engine_output"]
            )
            and add_binding
        )
        if add is not None and not local:
            reasons.append("Selection/postprocessor/engine/add call or squeeze-view witness is incomplete.")
        parent = events[origin] if origin is not None and origin < len(events) else None
        selection_origin_supported = (
            local if origin == ordinal else bool(parent and parent["_selection_origin_supported"])
        )
        previous_supported = (
            not requires_previous
            or previous_origin is not None
            and previous_origin < len(events)
            and events[previous_origin]["_selection_origin_supported"]
            and events[previous_origin]["_processor_supported"]
        )
        chain_origin_supported = selection_origin_supported and previous_supported
        effective = (
            observed_processor
            if origin == ordinal
            else None
            if parent is None
            else parent["observed_processor"]
        )
        selected_origin = (
            tick.get("selected") if origin == ordinal else None if parent is None else parent["_selected_ref"]
        )
        post_origin = pp_output if origin == ordinal else None if parent is None else parent["_post_ref"]
        processor_ok = (
            processor_supported if origin == ordinal else bool(parent and parent["_processor_supported"])
        )
        consumer_binding = (
            get_binding
            and relations["get_return_to_consumer"] is True
            and tokens["get_output"] is not None
            and tokens["get_output"] == tokens["requested"]
            and tick.get("requested") is not None
        )
        dispatch_binding = (
            relations["consumer_to_dispatch_target"] is True
            and tokens["requested"] is not None
            and tokens["requested"] == tokens["dispatch_return"]
            and tick.get("dispatch_return") is not None
        )
        complete = (
            supported
            and covered
            and chain_origin_supported
            and processor_ok
            and consumer_binding
            and dispatch_binding
            and tick["outcome"] == "sent"
            and tick["consumer_completed"]
            and tick["ack_count"] == 1
        )
        if not processor_ok:
            reasons.append(
                "Effective actual processor, converters, hooks or originating call are unsupported/incomplete."
            )
        if not complete:
            reasons.append(
                "Full observed selection-to-consumer chain is incomplete; no command acceptance is certified."
            )
        numeric_status = (
            "unknown"
            if not supported or not processor_ok
            else "error"
            if any(v is False or isinstance(v, np.bool_) and not v for v in numeric)
            else "pass"
            if numeric
            else "unknown"
        )
        events.append(
            {
                "tick_id": ordinal,
                "outcome": tick["outcome"],
                "completion": tick["completion"],
                "origin_tick_id": origin,
                "requires_previous_origin": requires_previous,
                "previous_origin_tick_id": previous_origin if requires_previous else None,
                "origin_supported": bool(chain_origin_supported),
                "binding_status": "pass" if complete else "unknown",
                "numeric_replay_status": numeric_status,
                "observed_processor": effective,
                "emitted_policy_action": emitted,
                "consumer_completed": tick["consumer_completed"],
                "ack_count": tick["ack_count"],
                "reason": "Observed native call, transformation and consumer argument chain."
                if complete
                else " ".join(reasons),
                "_selection_origin_supported": bool(local)
                if add is not None
                else bool(selection_origin_supported),
                "_selected_ref": selected_origin,
                "_post_ref": post_origin,
                "_processor_supported": processor_ok,
                "_role_refs": role_refs,
            }
        )
    return events, covered


def _states_values_equal(left, right, a, equal):
    return (
        len(left["buffer"]) == len(right["buffer"])
        and all(equal(a(x), a(y)) for x, y in zip(left["buffer"], right["buffer"], strict=False))
        and (
            (left["previous"] is right["previous"] is None)
            or left["previous"] is not None
            and right["previous"] is not None
            and equal(a(left["previous"]), a(right["previous"]))
        )
    )


def _states_equal(left, right, a, equal):
    return (
        _states_values_equal(left, right, a, equal)
        and left["index"] == right["index"]
        and left["emitted"] == right["emitted"]
        and left["previous_token"] == right["previous_token"]
        and left["buffer_tokens"] == right["buffer_tokens"]
    )


def _audit(
    data,
    arrays,
    hashes,
    root,
    supported,
    expected_processor,
    expected_action_names,
    expected_consumer_namespace,
    atol,
    rtol,
):
    expected_processor = _processor_contract(expected_processor)
    if expected_action_names is not None:
        expected_action_names = _names(expected_action_names)
    if expected_consumer_namespace is not None and not _text(expected_consumer_namespace):
        raise InputError("Expected consumer namespace must be bounded explicit text")
    if expected_processor is not None and len(expected_processor["mean"]) != len(data["action_names"]):
        raise InputError("Expected processor statistics width differs from captured declared order")
    events, covered = _analyze(data, arrays, supported, atol, rtol)
    numeric = (
        "error"
        if any(e["numeric_replay_status"] == "error" for e in events)
        else "pass"
        if all(e["numeric_replay_status"] == "pass" for e in events)
        else "unknown"
    )
    binding = "pass" if all(e["binding_status"] == "pass" for e in events) else "unknown"
    qualified = supported and covered and numeric == "pass" and binding == "pass"
    violations = []
    if qualified:
        if expected_action_names is not None and expected_action_names != data["action_names"]:
            violations.append(
                {"field": "action_names", "observed": data["action_names"], "expected": expected_action_names}
            )
        if (
            expected_consumer_namespace is not None
            and expected_consumer_namespace != data["consumer_namespace"]
        ):
            violations.append(
                {
                    "field": "consumer_namespace",
                    "observed": data["consumer_namespace"],
                    "expected": expected_consumer_namespace,
                }
            )
        if expected_processor is not None:
            for e in events:
                actual = e["observed_processor"]
                if (
                    actual is None
                    or actual["mode"] != expected_processor["mode"]
                    or any(
                        not np.allclose(actual[n], expected_processor[n], atol=atol, rtol=rtol)
                        for n in ("mean", "std")
                    )
                ):
                    violations.append(
                        {
                            "field": "effective_processor",
                            "tick_id": e["tick_id"],
                            "observed": actual,
                            "expected": expected_processor,
                        }
                    )
                    break
    explicit = any(
        v is not None for v in (expected_processor, expected_action_names, expected_consumer_namespace)
    )
    criterion = "unknown" if not qualified or not explicit else "error" if violations else "pass"
    artifacts = [{"path": str(root / n), "sha256": h} for n, h in sorted(hashes.items())]
    evidence = {"path": str(root / "capture.json"), "sha256": hashes["capture.json"], "pointer": "/ticks"}
    ticks = []
    for event in events:
        item = {k: v for k, v in event.items() if not k.startswith("_") and k != "origin_supported"}
        processor = item.get("observed_processor")
        if processor is not None:
            item["observed_processor"] = {
                "mode": processor["mode"],
                "mean": processor["mean"][:8],
                "std": processor["std"][:8],
            }
            item["processor_statistic_width"] = len(processor["mean"])
            item["processor_values_truncated"] = len(processor["mean"]) > 8
        item["evidence"] = {
            "path": str(root / "capture.json"),
            "sha256": hashes["capture.json"],
            "pointer": f"/ticks/{event['tick_id']}",
        }
        ticks.append(item)
    processors = [e["observed_processor"] for e in events if e["observed_processor"] is not None]
    uniform = bool(processors) and all(p == processors[0] for p in processors)
    return {
        "kind": "command_path_audit",
        "status": "error" if numeric == "error" else criterion,
        "binding_status": binding,
        "numeric_replay_status": numeric,
        "source_supported": supported,
        "coverage_supported": covered,
        "acceptance_contract": {
            "status": criterion,
            "expected_processor": expected_processor,
            "expected_action_names": expected_action_names,
            "expected_consumer_namespace": expected_consumer_namespace,
            "violations": violations,
            "evidence": [evidence],
            "scope": "Explicit effective processor and declared tensor-order/software-namespace acceptance; permits new action values and lawful consumer returns.",
        },
        "observed_processor": deepcopy(processors[0]) if uniform else None,
        "action_names": data["action_names"],
        "consumer_namespace": data["consumer_namespace"],
        "tick_count": len(events),
        "ticks": ticks,
        "query_lineage_status": "unknown",
        "recorded_action_status": "unknown",
        "physical_execution_status": "unknown",
        "tolerances": {"atol": atol, "rtol": rtol},
        "artifacts": artifacts,
        "scope": SCOPE,
    }


def audit_command_capture(
    capture,
    *,
    expected_processor=None,
    expected_action_names=None,
    expected_consumer_namespace=None,
    atol=1e-6,
    rtol=1e-5,
):
    """Check one immutable native command snapshot without Torch, policy or driver execution."""
    _tolerances(atol, rtol)
    root, data, arrays, hashes, supported = _load(capture)
    report = _audit(
        data,
        arrays,
        hashes,
        root,
        supported,
        expected_processor,
        expected_action_names,
        expected_consumer_namespace,
        atol,
        rtol,
    )
    _recheck(root, hashes)
    return report


def compare_command_captures(reference, candidate, *, atol=1e-6, rtol=1e-5):
    """Locate the first observed differing role at captured tick ordinal; no automatic fault inference."""
    _tolerances(atol, rtol)
    left = _load(reference)
    right = _load(candidate)
    analyses = [_analyze(d, a, s, atol, rtol)[0] for _, d, a, _, s in (left, right)]
    qualified = all(
        s and all(e["binding_status"] == "pass" and e["numeric_replay_status"] == "pass" for e in events)
        for (_, _, _, _, s), events in zip((left, right), analyses, strict=True)
    )
    context_differences = []
    for field in ("action_names", "consumer_namespace"):
        if left[1][field] != right[1][field]:
            context_differences.append(
                {"field": field, "reference": left[1][field], "candidate": right[1][field]}
            )
    for ordinal, (ae, be) in enumerate(zip(*analyses, strict=False)):
        if ae["observed_processor"] != be["observed_processor"]:
            context_differences.append(
                {
                    "field": "effective_processor",
                    "tick_ordinal": ordinal,
                    "reference": ae["observed_processor"],
                    "candidate": be["observed_processor"],
                }
            )
    if (
        left[1]["action_names"] != right[1]["action_names"]
        or left[1]["consumer_namespace"] != right[1]["consumer_namespace"]
    ):
        qualified = False
    first = None
    if qualified:
        for ordinal, (a_event, b_event) in enumerate(zip(*analyses, strict=False)):
            for stage, a_ref, b_ref in [
                ("selection", a_event["_selected_ref"], b_event["_selected_ref"]),
                ("postprocessor", a_event["_post_ref"], b_event["_post_ref"]),
                (
                    "engine_output",
                    a_event["_role_refs"]["engine_output"],
                    b_event["_role_refs"]["engine_output"],
                ),
                ("consumer_request", a_event["_role_refs"]["requested"], b_event["_role_refs"]["requested"]),
                ("consumer_return", a_event["_role_refs"]["returned"], b_event["_role_refs"]["returned"]),
            ]:
                x = None if a_ref is None else left[2][a_ref]
                y = None if b_ref is None else right[2][b_ref]
                same = (
                    x is y is None
                    or x is not None
                    and y is not None
                    and x.shape == y.shape
                    and np.allclose(x, y, atol=atol, rtol=rtol)
                )
                if not same:
                    first = {
                        "tick_ordinal": ordinal,
                        "stage": stage,
                        "reference": None if x is None else x.tolist(),
                        "candidate": None if y is None else y.tolist(),
                        "evidence": [
                            {
                                "path": str(bundle[0] / "capture.json"),
                                "sha256": bundle[3]["capture.json"],
                                "pointer": f"/ticks/{ordinal}",
                            }
                            for bundle in (left, right)
                        ],
                    }
                    break
            if first:
                break
    for bundle in (left, right):
        _recheck(bundle[0], bundle[3])
    context_preview = []
    for difference in context_differences[:4]:
        item = deepcopy(difference)
        for role in ("reference", "candidate"):
            value = item[role]
            if type(value) is list:
                item[role] = value[:8]
                item[role + "_truncated"] = len(value) > 8
            elif type(value) is dict:
                item[role] = {"mode": value["mode"], "mean": value["mean"][:8], "std": value["std"][:8]}
                item[role + "_truncated"] = len(value["mean"]) > 8
        context_preview.append(item)
    return {
        "kind": "command_path_comparison",
        "status": "unknown",
        "comparison_supported": qualified,
        "first_difference": first,
        "difference_is_fault": False,
        "captured_tick_counts": [len(e) for e in analyses],
        "context_differences": context_preview,
        "context_difference_count": len(context_differences),
        "context_differences_truncated": len(context_differences) > 4,
        "equal_over_compared_ticks": None if not qualified else first is None,
        "comparison_complete": qualified and len(analyses[0]) == len(analyses[1]),
        "alignment_basis": "Captured dispatch-tick ordinal only; no episode/time/physical correspondence asserted.",
        "scope": SCOPE,
    }
