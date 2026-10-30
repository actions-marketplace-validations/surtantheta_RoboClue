"""Hand-computed command chains; no native runtime, model or robot executes here."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from policy_doctor import command_path, command_rules
from policy_doctor.artifacts import InputError


@pytest.fixture
def packet(tmp_path, monkeypatch):
    def make(name="capture", *, selected=(0.25, -0.5), mean=(10.0, -20.0), returned=None):
        root = tmp_path / name
        root.mkdir()
        (root / "sources").mkdir()
        hashes = {}
        for source in command_rules.SOURCE_HASHES:
            raw = ("controlled source " + source).encode()
            (root / "sources" / source).write_bytes(raw)
            hashes[source] = hashlib.sha256(raw).hexdigest()
        monkeypatch.setattr(command_path, "SOURCE_HASHES", hashes)
        s = np.array([selected], dtype=np.float32)
        m = np.array(mean, dtype=np.float32)
        std = np.array([2.0, 4.0], dtype=np.float32)
        p = s * std + m
        e = p[0].copy()
        r = e.copy()
        ret = r.copy() if returned is None else np.array(returned, dtype=np.float32)
        arrays = {"s": s, "m": m, "std": std, "p": p, "e": e, "r": r, "ret": ret}
        state = {
            "previous": None,
            "previous_token": None,
            "buffer": [],
            "buffer_tokens": [],
            "index": 0,
            "emitted": False,
        }
        added = {
            "previous": "e",
            "previous_token": "o000004",
            "buffer": ["r"],
            "buffer_tokens": ["o000003"],
            "index": 0,
            "emitted": False,
        }
        after = deepcopy(added)
        after.update(index=1, emitted=True)
        pp = {
            "input": "s",
            "output": "p",
            "hooks_present": False,
            "converters_supported": True,
            "steps": [
                {
                    "index": 0,
                    "kind": "unnormalizer",
                    "input": "s",
                    "output": "p",
                    "configuration": {
                        "mode": "MEAN_STD",
                        "eps": 1e-8,
                        "stats": {"mean": "m", "std": "std"},
                        "device": "cpu",
                        "float_dtype": None,
                    },
                    "transform": {
                        "input": "s",
                        "output": "p",
                        "inverse": True,
                        "key": "action",
                        "stats": {"mean": "m", "std": "std"},
                    },
                },
                {
                    "index": 1,
                    "kind": "device",
                    "input": "p",
                    "output": "p",
                    "configuration": {"device": "cpu", "float_dtype": None},
                },
            ],
        }
        postview = {
            "base_token": "o000001",
            "shape": [1, 2],
            "stride": [2, 1],
            "storage_offset": 0,
            "element_size": 4,
        }
        engineview = {
            "base_token": "o000001",
            "shape": [2],
            "stride": [1],
            "storage_offset": 0,
            "element_size": 4,
        }
        tick = {
            "tick_id": 0,
            "completion": "normal",
            "outcome": "sent",
            "selected": "s",
            "postprocessor": pp,
            "engine_output": "e",
            "interpolation": {
                "multiplier": 1,
                "pre": state,
                "add": {"input": "e", "post": added},
                "get": {"output": "r", "pre": deepcopy(added), "post": after},
            },
            "requested": "r",
            "returned": "ret",
            "dispatch_return": "r",
            "consumer_called": True,
            "consumer_completed": True,
            "ack_count": 1,
            "reason": None,
            "bindings": {
                "tokens": {
                    "selected": "o000000",
                    "post_input": "o000000",
                    "post_output": "o000001",
                    "engine_output": "o000002",
                    "add_input": "o000002",
                    "get_output": "o000003",
                    "requested": "o000003",
                    "returned": "o000005",
                    "dispatch_return": "o000003",
                },
                "relations": {
                    n: True
                    for n in (
                        "selected_to_post_input",
                        "post_return_is_engine_action",
                        "engine_to_add_input",
                        "get_prebuffer_to_return",
                        "get_return_to_consumer",
                        "consumer_to_dispatch_target",
                    )
                },
                "post_to_engine_view": {
                    "post": postview,
                    "engine": engineview,
                    "same_live_storage": True,
                    "basis": "live_post_output_storage",
                },
            },
        }
        metadata = {
            "schema_version": 1,
            "kind": "command_path_capture",
            "domain": "software_consumer",
            "source": {
                "repository": command_rules.REPOSITORY,
                "revision": command_rules.REVISION,
                "files": {n: {"sha256": h} for n, h in hashes.items()},
            },
            "action_names": ["shoulder", "wrist"],
            "consumer_namespace": "generated/cpu-recipient",
            "coverage": {
                "target": "one_sync_engine_dispatch_on_owner_thread",
                "complete_target_call_history": True,
                "external_state_mutations": "untracked",
            },
            "ticks": [tick],
            "arrays": {n: {"shape": list(a.shape), "dtype": str(a.dtype)} for n, a in arrays.items()},
        }
        np.savez(root / "arrays.npz", **arrays)
        metadata["files"] = [
            {"path": "arrays.npz", "sha256": hashlib.sha256((root / "arrays.npz").read_bytes()).hexdigest()},
            *[{"path": "sources/" + n, "sha256": h} for n, h in hashes.items()],
        ]
        write(root, metadata)
        return root

    return make


def write(root, data):
    raw = json.dumps(data).encode()
    (root / "capture.json").write_bytes(raw)
    (root / "READY.json").write_text(
        json.dumps({"schema_version": 1, "manifest_sha256": hashlib.sha256(raw).hexdigest()})
    )


def change(root, mutation):
    data = json.loads((root / "capture.json").read_text())
    mutation(data)
    write(root, data)


def contract():
    return {
        "expected_processor": {"mode": "MEAN_STD", "mean": [10.0, -20.0], "std": [2.0, 4.0]},
        "expected_action_names": ["shoulder", "wrist"],
        "expected_consumer_namespace": "generated/cpu-recipient",
    }


def test_supported_chain_replays_actual_stats_and_native_clone(packet):
    root = packet()
    report = command_path.audit_command_capture(root, **contract())
    assert report["status"] == "pass"
    assert report["binding_status"] == "pass" and report["numeric_replay_status"] == "pass"
    assert report["ticks"][0]["origin_tick_id"] == 0
    assert report["recorded_action_status"] == "unknown" and report["physical_execution_status"] == "unknown"
    assert (
        command_path.audit_command_capture(root)["status"] == "unknown"
    )  # Acceptance needs an operator contract.


def test_same_selection_changed_processor_localized_without_calling_clipping_a_fault(packet):
    reference = packet("ref")
    processor = packet("processor", mean=(13.0, -20.0))
    returned = packet("returned", returned=(10.5, -15.0))
    comparison = command_path.compare_command_captures(reference, processor)
    assert comparison["first_difference"]["stage"] == "postprocessor"
    assert comparison["difference_is_fault"] is False
    report = command_path.audit_command_capture(processor, **contract())
    assert report["status"] == "error" and report["acceptance_contract"]["status"] == "error"
    assert report["numeric_replay_status"] == "pass"
    assert (
        command_path.compare_command_captures(reference, returned)["first_difference"]["stage"]
        == "consumer_return"
    )
    assert command_path.audit_command_capture(returned, **contract())["status"] == "pass"


def test_same_numbers_with_unobserved_clone_do_not_gain_association(packet):
    root = packet()
    change(root, lambda d: d["ticks"][0]["bindings"]["tokens"].update(requested="o000008"))
    report = command_path.audit_command_capture(root, **contract())
    assert report["status"] == "unknown" and report["binding_status"] == "unknown"


def test_arithmetic_error_cannot_be_saved_as_contract_error(packet):
    root = packet()
    change(root, lambda d: d["ticks"][0]["postprocessor"]["steps"][0]["transform"].update(output="s"))
    report = command_path.audit_command_capture(root, **contract())
    assert report["status"] == "error"
    assert report["numeric_replay_status"] == "error" and report["acceptance_contract"]["status"] == "unknown"


def test_unknown_custom_hook_or_coverage_never_certifies_contract(packet):
    for name, mutate in [
        ("hooks", lambda d: d["ticks"][0]["postprocessor"].update(hooks_present=True)),
        ("coverage", lambda d: d["coverage"].update(complete_target_call_history=False)),
        ("source", lambda d: d["source"].update(revision="unsupported")),
    ]:
        root = packet(name)
        change(root, mutate)
        assert command_path.audit_command_capture(root, **contract())["status"] == "unknown"


def test_selection_difference_is_first_observation_and_not_an_intrinsic_fault(packet):
    a = packet("a")
    b = packet("b", selected=(0.5, 0.75))
    result = command_path.compare_command_captures(a, b)
    assert result["first_difference"]["stage"] == "selection"
    assert command_path.audit_command_capture(b, **contract())["status"] == "pass"


def test_hash_tamper_and_symlink_refused(packet, tmp_path):
    root = packet()
    (root / "arrays.npz").write_bytes(b"changed")
    with pytest.raises(InputError, match="hash"):
        command_path.audit_command_capture(root)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(InputError, match="symlink"):
        command_path.audit_command_capture(alias)


def arrays_update(root, **changed):
    with np.load(root / "arrays.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    arrays.update(changed)
    np.savez(root / "arrays.npz", **arrays)

    def mutate(d):
        d["arrays"] = {k: {"shape": list(a.shape), "dtype": str(a.dtype)} for k, a in arrays.items()}
        for e in d["files"]:
            if e["path"] == "arrays.npz":
                e["sha256"] = hashlib.sha256((root / "arrays.npz").read_bytes()).hexdigest()

    change(root, mutate)


def test_legal_custom_hook_change_is_unknown_without_a_fabricated_numeric_fault(packet):
    root = packet()
    new_input = np.array([[1.0, 2.0]], dtype=np.float32)
    p = np.array([[12.0, -12.0]], dtype=np.float32)
    arrays_update(root, h=new_input, p=p, e=p[0].copy(), r=p[0].copy(), ret=p[0].copy())

    def mutate(d):
        pp = d["ticks"][0]["postprocessor"]
        pp["hooks_present"] = True
        pp["steps"][0]["input"] = "h"
        pp["steps"][0]["transform"]["input"] = "h"

    change(root, mutate)
    report = command_path.audit_command_capture(root, **contract())
    assert report["status"] == "unknown" and report["numeric_replay_status"] == "unknown"


def test_missing_actual_used_stats_stays_unknown(packet):
    root = packet()
    change(root, lambda d: d["ticks"][0]["postprocessor"]["steps"][0]["transform"].update(stats={}))
    report = command_path.audit_command_capture(root, **contract())
    assert report["status"] == "unknown"


def test_view_shape_metadata_refuses_boolean_dimensions(packet):
    root = packet()
    change(root, lambda d: d["ticks"][0]["bindings"]["post_to_engine_view"]["post"].update(shape=[True, 2]))
    with pytest.raises(InputError, match="view layout"):
        command_path.audit_command_capture(root, **contract())


def continuation(tick, *, output="r", buffer_ref="r", token="o000003"):
    result = deepcopy(tick)
    result.update(
        tick_id=1,
        selected=None,
        postprocessor=None,
        engine_output=None,
        requested=output,
        returned=output,
        dispatch_return=output,
    )
    result["bindings"]["tokens"] = {k: None for k in command_path.ROLES}
    result["bindings"]["tokens"].update(
        get_output=token, requested=token, returned=token, dispatch_return=token
    )
    result["bindings"]["relations"] = {k: None for k in command_path.RELATIONS}
    result["bindings"]["relations"].update(
        get_prebuffer_to_return=True, get_return_to_consumer=True, consumer_to_dispatch_target=True
    )
    result["bindings"]["post_to_engine_view"] = None
    pre = deepcopy(tick["interpolation"]["get"]["post"])
    pre.update(index=0, buffer=[buffer_ref])
    post = deepcopy(pre)
    post.update(index=1, emitted=True)
    result["interpolation"] = {
        "multiplier": 1,
        "pre": pre,
        "add": None,
        "get": {"pre": deepcopy(pre), "post": post, "output": output},
    }
    return result


def test_mutated_retained_buffer_cannot_inherit_old_processor_contract(packet):
    root = packet()
    arrays_update(root, changed=np.array([11.5, -22.0], dtype=np.float32))
    change(
        root, lambda d: d["ticks"].append(continuation(d["ticks"][0], output="changed", buffer_ref="changed"))
    )
    report = command_path.audit_command_capture(root, **contract())
    assert report["status"] == "unknown"
    assert report["ticks"][1]["origin_tick_id"] is None
    assert report["ticks"][1]["numeric_replay_status"] != "error"


def test_unknown_prior_interpolation_origin_cannot_certify_whole_intermediate_chain(packet):
    root = packet()

    def mutate(d):
        t = d["ticks"][0]
        ip = t["interpolation"]
        ip["multiplier"] = 2
        ip["pre"].update(previous="e", previous_token="o000020")
        ip["add"]["post"].update(buffer=["r", "r"], buffer_tokens=["o000003", "o000030"])
        ip["get"]["pre"] = deepcopy(ip["add"]["post"])
        ip["get"]["post"] = deepcopy(ip["get"]["pre"])
        ip["get"]["post"].update(index=1, emitted=False)

    change(root, mutate)
    report = command_path.audit_command_capture(root, **contract())
    assert report["status"] == "unknown" and report["ticks"][0]["emitted_policy_action"] is False


def test_equal_numeric_outputs_keep_effective_parameter_difference(packet):
    a = packet("a", selected=(0.0, 0.0), mean=(0.0, 0.0))
    b = packet("b", selected=(0.0, 0.0), mean=(0.0, 0.0))
    arrays_update(b, std=np.array([3.0, 5.0], dtype=np.float32))
    result = command_path.compare_command_captures(a, b)
    assert result["first_difference"] is None and result["equal_over_compared_ticks"] is True
    assert result["context_differences"][0]["field"] == "effective_processor"
    assert result["context_difference_count"] == 1


def test_successful_none_return_is_a_completed_software_call_not_feedback(packet):
    root = packet()

    def mutate(d):
        d["ticks"][0]["returned"] = None
        d["ticks"][0]["bindings"]["tokens"]["returned"] = None

    change(root, mutate)
    report = command_path.audit_command_capture(root, **contract())
    assert report["status"] == "pass" and report["ticks"][0]["consumer_completed"]
    assert report["physical_execution_status"] == "unknown"


@pytest.mark.parametrize("number", [True, 10**400, float("inf")])
def test_expected_statistics_refuse_boolean_overflow_and_nonfinite(packet, number):
    root = packet()
    expected = contract()
    expected["expected_processor"]["mean"][0] = number
    with pytest.raises(InputError, match="finite numeric vectors"):
        command_path.audit_command_capture(root, **expected)


def test_maximum_tick_and_coordinate_report_is_bounded(packet):
    root = packet()
    width = 256
    a = np.zeros((1, width), dtype=np.float32)
    v = np.zeros(width, dtype=np.float32)
    arrays_update(root, s=a, m=v, std=np.ones(width, dtype=np.float32), p=a, e=v, r=v, ret=v)

    def mutate(d):
        d["action_names"] = [f"coordinate{i}" for i in range(width)]
        prototype = d["ticks"][0]
        prototype["bindings"]["post_to_engine_view"]["post"].update(shape=[1, width], stride=[width, 1])
        prototype["bindings"]["post_to_engine_view"]["engine"].update(shape=[width])
        d["ticks"] = []
        for i in range(64):
            t = deepcopy(prototype)
            t["tick_id"] = i

            def tokens(node, i=i):
                if isinstance(node, dict):
                    for k, value in node.items():
                        node[k] = tokens(value)
                elif isinstance(node, list):
                    return [tokens(value) for value in node]
                elif isinstance(node, str) and node.startswith("o") and len(node) == 7 and node[1:].isdigit():
                    return f"o{int(node[1:]) + i * 10:06d}"
                return node

            d["ticks"].append(tokens(t))

    change(root, mutate)
    report = command_path.audit_command_capture(
        root, expected_processor={"mode": "MEAN_STD", "mean": v.tolist(), "std": np.ones(width).tolist()}
    )
    assert report["status"] == "pass" and report["tick_count"] == 64
    assert len(json.dumps(report, allow_nan=False).encode()) <= 192 * 1024
    assert report["ticks"][0]["processor_values_truncated"]
