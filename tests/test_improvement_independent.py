"""Independent critique controls; generated responses only, never Hub traffic."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

from policy_doctor import hub
from policy_doctor.artifacts import InputError


def test_metadata_redirect_cannot_select_weights_at_the_same_pinned_revision(monkeypatch):
    """A pinned commit appearing in a URL alone does not make it metadata-only."""
    sha = "1" * 40
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            if len(calls) > 1:
                raise AssertionError("A metadata redirect must not initiate a weight request")
            raise HTTPError(request.full_url, 302, "redirect", {
                "Location": f"https://huggingface.co/org/repo/resolve/{sha}/model.safetensors"
            }, None)

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    with pytest.raises(InputError):
        hub._request(f"https://huggingface.co/org/repo/resolve/{sha}/config.json",
                     limit=1024, required_sha=sha)
    assert len(calls) == 1


def test_metadata_body_timeout_is_an_unknown_transport_failure(monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, count):
            raise TimeoutError("Generated slow metadata body")

    class Opener:
        def open(self, request, timeout):
            return Response()

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    with pytest.raises(hub.HubUnavailable):
        hub._request("https://huggingface.co/api/models/org/repo/revision/main", limit=1024)


def test_documented_policy_first_use_ends_in_evidence_and_a_next_step(tmp_path):
    """Exercise the two README commands with no supplied capture or runtime."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    def run(*args):
        return subprocess.run([sys.executable, "-m", "policy_doctor", *args], cwd=tmp_path,
                              env=env, capture_output=True, text=True, timeout=30)

    created = run("demo", "./robot-example", "--kind", "policy")
    assert created.returncode == 0, created.stderr
    result = run("check", "./robot-example/checkpoint", "--dataset", "./robot-example/dataset")
    assert result.returncode == 1, result.stderr
    assert len(result.stdout.encode()) < 2048
    assert "action.processor_order_names" in result.stdout
    assert "shoulder" in result.stdout and "gripper" in result.stdout
    assert "/steps/" in result.stdout
    assert "next" in result.stdout.lower()
    assert any(word in result.stdout.lower() for word in ("restore", "reorder", "correct"))

    healthy = run("check", "./robot-example/checkpoint", "--dataset", "./robot-example/corrected_dataset")
    assert healthy.returncode == 3, healthy.stderr
    assert "unknown" in healthy.stdout.lower()


@pytest.mark.parametrize("names", [["state"], {"axes": None}])
@pytest.mark.parametrize("actual_width", [2, 3])
def test_unsupported_vector_name_encoding_keeps_independent_width_check(monkeypatch, names, actual_width):
    """A strict local parser must not turn an unmodeled native encoding into a bug."""
    sha = "1" * 40
    config = {"type": "act", "input_features": {
        "observation.state": {"type": "STATE", "shape": [2]}}}
    info = {"fps": 30, "features": {
        "observation.state": {"dtype": "float32", "shape": [actual_width], "names": names}}}

    def request(url, *, limit, required_sha=None):
        if "/api/models/" in url:
            value = {"id": "org/model", "sha": sha, "private": False,
                     "siblings": [{"rfilename": "config.json"}]}
        elif "/api/datasets/" in url:
            value = {"id": "org/data", "sha": sha, "private": False,
                     "siblings": [{"rfilename": "meta/info.json"}]}
        elif url.endswith("/config.json"):
            value = config
        elif url.endswith("/meta/info.json"):
            value = info
        else:
            raise AssertionError(f"Unexpected generated network request: {url}")
        return json.dumps(value).encode()

    monkeypatch.setattr(hub, "_request", request)
    report = hub.check_checkpoint("hf://org/model", "hf://org/data", fetch=True)
    assert report["status"] == ("unknown" if actual_width == 2 else "error")
    assert report["counts"]["error"] == (0 if actual_width == 2 else 1)
    assert report["counts"]["unknown"] > 0
    assert len(report["provenance"]) == 2
    # Unknown coordinate names must not hide an independently answerable width.
    shape = next(f for f in report["findings"]
                 if f["code"] == "dataset.feature_shape"
                 and f["status"] == ("pass" if actual_width == 2 else "error"))
    dataset_sha = hashlib.sha256(json.dumps(info).encode()).hexdigest()
    witnesses = [entry for entry in shape["evidence"]
                 if entry["path"] == f"hf://org/data@{sha}/meta/info.json"
                 and entry["pointer"] == "/features/observation.state"
                 and entry["sha256"] == dataset_sha]
    assert len(witnesses) == 1
    original = witnesses[0].get("source_value", witnesses[0]["value"])
    assert original["names"] == names
    assert original["shape"] == [actual_width]
