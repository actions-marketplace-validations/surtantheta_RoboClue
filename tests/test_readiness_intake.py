"""Intake permissions and original-byte association, independent of readiness rules."""

import hashlib
import io
import json
import struct
from urllib.error import HTTPError

import numpy as np
import pytest
from safetensors.numpy import save

from policy_doctor import hub
from policy_doctor import readiness_intake as intake
from policy_doctor.artifacts import InputError, load_checkpoint

SHA = "a" * 40
SOURCE = "hf://org/policy"


def encoded(value):
    return json.dumps(value, separators=(",", ":")).encode()


@pytest.fixture
def repository(monkeypatch):
    feature = {"observation.state": {"type": "STATE", "shape": [2]}}
    files = {
        "config.json": encoded({"type": "act", "input_features": feature}),
        "train_config.json": encoded({"dataset": {"repo_id": "org/data", "revision": "b" * 40}}),
        "policy_preprocessor.json": encoded({"steps": [{"registry_name": "normalizer_processor",
            "config": {"features": feature, "norm_map": {"STATE": "MEAN_STD"}}, "state_file": "pre.safetensors"}]}),
        "pre.safetensors": save({"observation.state.mean": np.array([1, 2], dtype=np.float32),
                                 "observation.state.std": np.array([3, 4], dtype=np.float32)}),
        "model.safetensors": b"must never be requested",
    }
    calls = []

    def api():
        return {"id": "org/policy", "sha": SHA, "private": False, "gated": False,
                "siblings": [{"rfilename": name, "size": len(body), "blobId": hashlib.sha1(
                    b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()} for name, body in files.items()]}

    def request(url, *, limit, required_sha=None):
        calls.append(url)
        if "/api/models/" in url:
            assert required_sha is None
            body = encoded(api())
        else:
            assert required_sha == SHA and f"/resolve/{SHA}/" in url
            body = files[url.split(f"/resolve/{SHA}/")[1]]
        assert len(body) <= limit
        return body

    monkeypatch.setattr(hub, "_request", request)
    return files, calls, api


def test_declared_stats_are_native_readable_and_training_reference_is_retained(repository):
    files, calls, _ = repository
    budget = intake.ReadinessIntakeBudget()
    with intake.resolve_readiness_snapshot(SOURCE, fetch=True, budget=budget, processor_state=True) as view:
        assert (view.root / "train_config.json").read_bytes() == files["train_config.json"]
        cp = load_checkpoint(view.root)
        assert cp.pre[0]["stats"]["observation.state"]["mean"] == [1, 2]
        assert cp.pre[0]["stats"]["observation.state"]["std"] == [3, 4]
        assert not view.omitted_states
        assert all(row["listed_content_id_kind"] == "git_blob_sha1" for row in view.provenance["files"])
        assert not (view.root / "model.safetensors").exists()
    assert not any(url.endswith("model.safetensors") for url in calls)
    assert budget.used_bytes == budget.successful_body_bytes
    assert budget.logical_requests == 5


def test_new_permission_does_not_expand_legacy_json_tools(repository):
    _, calls, _ = repository
    with hub.resolve_hub_snapshot(SOURCE, fetch=True) as view:
        assert not (view.root / "train_config.json").exists()
        assert view.omitted_states == {"policy_preprocessor.json": ["/steps/0"]}
    assert not any(url.endswith(".safetensors") or url.endswith("train_config.json") for url in calls)


def test_native_whole_dataset_statistics_are_valid_for_action_postprocessor(repository):
    files, _, _ = repository
    files["policy_postprocessor.json"] = encoded({"steps": [{"registry_name": "unnormalizer_processor",
        "config": {"features": {"action": {"type": "ACTION", "shape": [2]}},
                   "norm_map": {"ACTION": "MEAN_STD"}}, "state_file": "post.safetensors"}]})
    files["post.safetensors"] = save({
        "action.mean": np.array([0, 0], dtype=np.float32),
        "action.std": np.array([1, 1], dtype=np.float32),
        "observation.state.mean": np.array([1, 2], dtype=np.float32),
        "observation.state.std": np.array([3, 4], dtype=np.float32),
        "observation.images.camera1.mean": np.zeros((3, 1, 1), dtype=np.float32),
        **{key + ".mean": np.array([0], dtype=np.float32)
           for key in ("episode_index", "frame_index", "index", "task_index", "timestamp", "next.reward")},
    })
    with intake.resolve_readiness_snapshot(SOURCE, fetch=True, processor_state=True,
                                           budget=intake.ReadinessIntakeBudget()) as view:
        cp = load_checkpoint(view.root)
        assert cp.post[0]["stats"]["action"]["std"] == [1, 1]
        assert cp.post[0]["stats"]["timestamp"]["mean"] == [0]
        assert not view.omitted_states


def test_withheld_and_missing_at_pin_are_distinct(repository):
    files, _, _ = repository
    with intake.resolve_readiness_snapshot(SOURCE, fetch=True, budget=intake.ReadinessIntakeBudget()) as view:
        assert view.omitted_states == {"policy_preprocessor.json": ["/steps/0"]}
        assert not (view.root / "pre.safetensors").exists()
    del files["pre.safetensors"]
    with intake.resolve_readiness_snapshot(SOURCE, fetch=True, budget=intake.ReadinessIntakeBudget()) as view:
        assert view.omitted_states == {}
        assert view.provenance["omitted_states"][0]["listed_at_revision"] is False


def test_unsupported_dtype_preserves_json_and_explicit_unknown_context(repository):
    files, _, _ = repository
    header = encoded({"observation.state.mean": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}})
    files["pre.safetensors"] = struct.pack("<Q", len(header)) + header + b"\0" * 4
    with intake.resolve_readiness_snapshot(SOURCE, fetch=True, processor_state=True,
                                           budget=intake.ReadinessIntakeBudget()) as view:
        assert (view.root / "config.json").exists()
        assert not (view.root / "pre.safetensors").exists()
        assert view.omitted_states == {"policy_preprocessor.json": ["/steps/0"]}
        row = view.provenance["files"][-1]
        assert row["numerical_assessment"] == "unknown" and row["retained_in_view"] is False
        assert "BF16" in row["reason"]


def test_parameter_keys_are_not_interpreted_as_statistics(repository):
    files, _, _ = repository
    files["pre.safetensors"] = save({"layer.weight": np.zeros((2, 2), dtype=np.float32)})
    with pytest.raises(InputError, match="outside its declared"):
        with intake.resolve_readiness_snapshot(SOURCE, fetch=True, processor_state=True,
                                               budget=intake.ReadinessIntakeBudget()):
            pass


def test_model_parameter_filename_is_refused_before_download(repository):
    files, calls, _ = repository
    processor = json.loads(files["policy_preprocessor.json"])
    processor["steps"][0]["state_file"] = "model.safetensors"
    files["policy_preprocessor.json"] = encoded(processor)
    with pytest.raises(InputError, match="Model parameter filenames"):
        with intake.resolve_readiness_snapshot(SOURCE, fetch=True, processor_state=True,
                                               budget=intake.ReadinessIntakeBudget()):
            pass
    assert not any(url.endswith("model.safetensors") for url in calls)


def test_body_must_match_the_same_pinned_listing(repository, monkeypatch):
    _, _, _ = repository
    original = hub._request

    def corrupt(url, **kwargs):
        body = original(url, **kwargs)
        return body.replace(b'"act"', b'"bad"') if url.endswith("config.json") else body

    monkeypatch.setattr(hub, "_request", corrupt)
    with pytest.raises(InputError, match="content identity"):
        with intake.resolve_readiness_snapshot(SOURCE, fetch=True, budget=intake.ReadinessIntakeBudget()):
            pass


@pytest.mark.parametrize("kwargs", [{"fetch": False}, {"fetch": 1}, {"processor_state": 1},
                                   {"repo_type": "dataset", "processor_state": True}])
def test_permissions_are_validated_before_any_http(repository, kwargs):
    _, calls, _ = repository
    args = {"fetch": True, "budget": intake.ReadinessIntakeBudget(), **kwargs}
    with pytest.raises(InputError):
        with intake.resolve_readiness_snapshot(SOURCE, **args):
            pass
    assert not calls


@pytest.mark.parametrize("kwargs", [{"max_bytes": True}, {"max_bytes": 0}, {"max_bytes": 65 * 1024 ** 2},
                                   {"max_files": True}, {"max_files": 0}, {"max_files": 65}])
def test_budget_bounds_are_strict(kwargs):
    with pytest.raises(InputError):
        intake.ReadinessIntakeBudget(**kwargs)


def test_failed_read_reservation_cannot_be_reused(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs["limit"])
        raise hub.HubUnavailable("interrupted")

    monkeypatch.setattr(hub, "_request", fail)
    budget = intake.ReadinessIntakeBudget(max_bytes=100, max_files=3)
    with pytest.raises(hub.HubUnavailable):
        budget.request("https://huggingface.co/test", limit=100)
    assert budget.used_bytes == 100 and calls == [99]
    with pytest.raises(InputError, match="aggregate"):
        budget.request("https://huggingface.co/test", limit=100)
    assert calls == [99] and budget.successful_body_bytes == 0


class Response(io.BytesIO):
    status = 200

    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}
        self.sizes = []

    def read1(self, n):
        self.sizes.append(n)
        return self.read(n)


def opener_for(monkeypatch, events):
    calls = []

    class Opener:
        def open(self, request, **kwargs):
            calls.append(request)
            event = events.pop(0)
            if isinstance(event, Exception):
                raise event
            return event

    monkeypatch.setattr(hub, "_opener", Opener)
    return calls


def test_pinned_lfs_statistics_follow_one_verified_https_cdn_hop(monkeypatch):
    data = b"original statistics bytes"
    digest = hashlib.sha256(data).hexdigest()
    redirect = HTTPError("https://huggingface.co/x", 302, "redirect",
                         {"Location": "https://cas-bridge.xethub.hf.co/object?bounded=1", "X-Linked-Etag": digest}, None)
    calls = opener_for(monkeypatch, [redirect, Response(data, {"Content-Length": str(len(data))})])
    result = intake._lfs_request(f"https://huggingface.co/org/policy/resolve/{SHA}/pre.safetensors",
                                limit=len(data), required_sha=SHA, sha256=digest)
    assert result == data and len(calls) == 2
    assert all(not request.has_header("Authorization") for request in calls)


@pytest.mark.parametrize("location", ["http://cas-bridge.xethub.hf.co/object", "https://evil.example/object",
                                      "https://user@cas-bridge.xethub.hf.co/object", "https://cas-bridge.xethub.hf.co:80/object"])
def test_lfs_refuses_redirect_before_contacting_unapproved_target(monkeypatch, location):
    redirect = HTTPError("https://huggingface.co/x", 302, "redirect", {"Location": location}, None)
    calls = opener_for(monkeypatch, [redirect])
    with pytest.raises(InputError, match="redirect"):
        intake._lfs_request(f"https://huggingface.co/org/policy/resolve/{SHA}/pre.safetensors",
                            limit=4, required_sha=SHA, sha256="0" * 64)
    assert len(calls) == 1


def test_lfs_unknown_length_oversize_reads_only_cap_plus_one(monkeypatch):
    response = Response(b"x" * 100)
    opener_for(monkeypatch, [response])
    with pytest.raises(InputError, match="body bound"):
        intake._lfs_request(f"https://huggingface.co/org/policy/resolve/{SHA}/pre.safetensors",
                            limit=3, required_sha=SHA, sha256="0" * 64)
    assert response.sizes == [4]


def test_lfs_declared_length_must_match_eof(monkeypatch):
    data = b"full"
    opener_for(monkeypatch, [Response(data, {"Content-Length": "3"})])
    with pytest.raises(InputError, match="declared HTTP length"):
        intake._lfs_request(f"https://huggingface.co/org/policy/resolve/{SHA}/pre.safetensors",
                            limit=4, required_sha=SHA, sha256=hashlib.sha256(data).hexdigest())


def test_lfs_body_failure_is_controlled_and_budget_stays_charged(monkeypatch):
    class Broken(Response):
        def read1(self, n):
            raise OSError("connection lost")

    opener_for(monkeypatch, [Broken(b"")])
    budget = intake.ReadinessIntakeBudget(max_bytes=100)
    with pytest.raises(hub.HubUnavailable, match="could not be read"):
        budget.request(f"https://huggingface.co/org/policy/resolve/{SHA}/pre.safetensors",
                       limit=99, required_sha=SHA, lfs_sha256="0" * 64)
    assert budget.used_bytes == 100 and budget.successful_body_bytes == 0
