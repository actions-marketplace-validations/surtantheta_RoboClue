"""Metadata transport tests use generated public-API responses; no network."""

import json

import pytest

from policy_doctor import hub
from policy_doctor.artifacts import InputError

SHA = "1" * 40
CONFIG = {"type": "act", "input_features": {"observation.state": {"type": "STATE", "shape": [2]}},
          "output_features": {"action": {"type": "ACTION", "shape": [2]}}}
INFO = {"fps": 30, "features": {"observation.state": {"dtype": "float32", "shape": [3]},
                              "action": {"dtype": "float32", "shape": [2]}}}


@pytest.fixture
def network(monkeypatch):
    calls = []
    files = json.loads(json.dumps({"config.json": CONFIG, "meta/info.json": INFO}))

    def request(url, *, limit, required_sha=None):
        calls.append((url, limit, required_sha))
        if "/api/" in url:
            kind = "model" if "/models/" in url else "dataset"
            entries = list(files) if kind == "model" else ["meta/info.json"]
            return json.dumps({"id": "org/repo", "sha": SHA, "private": False,
                               "siblings": [{"rfilename": f} for f in entries]}).encode()
        key = url.split("/resolve/" + SHA + "/")[1]
        return json.dumps(files[key]).encode()

    monkeypatch.setattr(hub, "_request", request)
    return calls, files


def test_no_opt_in_never_calls_network(network):
    calls, _ = network
    with pytest.raises(InputError, match="fetch"):
        hub.check_checkpoint("hf://org/repo")
    assert calls == []


@pytest.mark.parametrize("source", ["https://huggingface.co/org/repo", "hf://org/repo/extra",
    "hf://org/../repo", "hf://org/repo?x=1", "hf://org/repo#x", "hf://a%2fb/repo",
    "hf://user@host/repo", "hf://org/rep..o"])
def test_unsafe_ids_fail_before_transport(source, network):
    calls, _ = network
    with pytest.raises(InputError):
        hub.check_checkpoint(source, fetch=True)
    assert calls == []


@pytest.mark.parametrize("revision", ["../main", "main?x=1", "refs/pr/1", "a%2fb", "", "x" * 129])
def test_unsafe_revisions_fail_before_transport(revision, network):
    calls, _ = network
    with pytest.raises(InputError):
        hub.check_checkpoint("hf://org/repo", fetch=True, revision=revision)
    assert calls == []


def test_immutable_json_only_fetch_and_feature_mismatch(network):
    calls, files = network
    files["model.safetensors"] = {"never": "fetch"}
    result = hub.check_checkpoint("hf://org/repo", "hf://org/repo", fetch=True)
    assert result["kind"] == "checkpoint_check"
    assert result["status"] == "error"
    assert any(f["code"] == "dataset.feature_shape" and f["status"] == "error" for f in result["findings"])
    assert all("model.safetensors" not in c[0] for c in calls)
    downloads = [c for c in calls if "/resolve/" in c[0]]
    assert downloads and all("/resolve/" + SHA + "/" in c[0] and c[2] == SHA for c in downloads)
    assert len(result["provenance"]) == 2
    assert result["provenance"][0]["resolved_revision"] == SHA
    assert "policy-doctor-hub-" not in json.dumps(result)
    for p in result["provenance"]:
        assert p["downloaded_bytes"] == sum(f["size_bytes"] for f in p["files"])
        assert all(len(f["sha256"]) == 64 for f in p["files"])


def test_withheld_listed_state_is_unknown_not_missing_error(network):
    _, files = network
    step = {"registry_name": "normalizer_processor", "config": {
        "features": {"action": {"type": "ACTION", "shape": [2]}}, "norm_map": {"ACTION": "MEAN_STD"}},
        "state_file": "policy_preprocessor_step_0.safetensors"}
    files["policy_preprocessor.json"] = {"steps": [step]}
    files["policy_preprocessor_step_0.safetensors"] = {"never": "fetch"}
    result = hub.check_checkpoint("hf://org/repo", fetch=True)
    assert any(f["code"] == "processor.state_uninspected" and f["status"] == "unknown"
               for f in result["findings"])
    assert not any(f["code"] == "processor.state_missing" for f in result["findings"])
    assert not any(f["code"] == "normalization.statistics" and f["status"] == "error"
                   for f in result["findings"])


def test_genuine_absent_state_reference_remains_error(network):
    _, files = network
    files["policy_preprocessor.json"] = {"steps": [{"registry_name": "normalizer_processor",
         "config": {}, "state_file": "absent.safetensors"}]}
    result = hub.check_checkpoint("hf://org/repo", fetch=True)
    assert any(f["code"] == "processor.state_missing" and f["status"] == "error"
               for f in result["findings"])


def test_unmodeled_padding_does_not_claim_native_incompatibility(network):
    _, files = network
    files["policy_preprocessor.json"] = {"steps": [{"registry_name": "pad_state_processor",
                                                     "config": {"target_size": 2}}]}
    result = hub.check_checkpoint("hf://org/repo", "hf://org/repo", fetch=True)
    mismatches = [f for f in result["findings"] if f["code"] == "dataset.feature_shape"
                  and "observation.state" in f["message"]]
    assert len(mismatches) == 1 and mismatches[0]["status"] == "unknown"
    assert "not modeled" in mismatches[0]["message"]


def test_unsafe_state_reference_rejected(network):
    _, files = network
    files["policy_preprocessor.json"] = {"steps": [{"registry_name": "normalizer_processor",
         "config": {}, "state_file": "https://evil.test/a.safetensors"}]}
    with pytest.raises(InputError, match="reference"):
        hub.check_checkpoint("hf://org/repo", fetch=True)


def test_requested_commit_drift_refused(network):
    with pytest.raises(InputError, match="revision"):
        hub.check_checkpoint("hf://org/repo", fetch=True, revision="2" * 40)


def test_duplicate_json_rejected(monkeypatch):
    monkeypatch.setattr(hub, "_request", lambda *a, **k: b'{"sha":"a","sha":"b"}')
    with pytest.raises(InputError, match="Duplicate"):
        hub.check_checkpoint("hf://org/repo", fetch=True)


def test_private_api_refused(monkeypatch):
    monkeypatch.setattr(hub, "_request", lambda *a, **k: json.dumps({
        "id": "org/repo", "sha": SHA, "private": True, "siblings": []}).encode())
    with pytest.raises(InputError, match="public"):
        hub.check_checkpoint("hf://org/repo", fetch=True)


def test_both_sources_validated_before_first_network(network):
    calls, _ = network
    with pytest.raises(InputError):
        hub.check_checkpoint("hf://org/repo", "hf://a/../b", fetch=True)
    assert calls == []


@pytest.mark.parametrize("url", ["http://huggingface.co/a", "https://evil.test/a",
    "https://huggingface.co.evil.test/a", "https://127.0.0.1/a", "https://huggingface.co:444/a",
    "https://user:pw@huggingface.co/a", "https://huggingface.co/a#frag"])
def test_transport_rejects_unsafe_target(url):
    with pytest.raises(InputError):
        hub._validate_url(url)


def test_redirect_checked_before_next_request(monkeypatch):
    from urllib.error import HTTPError

    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            raise HTTPError(request.full_url, 302, "redirect", {"Location": "https://evil.test/x"}, None)

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    with pytest.raises(InputError, match="HTTPS"):
        hub._request("https://huggingface.co/api/models/org/repo/revision/main", limit=1024)
    assert len(calls) == 1


@pytest.mark.parametrize("header", ["99999999", None])
def test_transport_size_bound_with_and_without_length(monkeypatch, header):
    class Response:
        status = 200
        headers = {} if header is None else {"Content-Length": header}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, n):
            return b"x" * n

    class Opener:
        def open(self, request, timeout):
            return Response()

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    with pytest.raises(InputError, match="bound"):
        hub._request("https://huggingface.co/api/models/org/repo/revision/main", limit=1024)


def test_local_path_keeps_network_off(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))

    def reject(*args, **kwargs):
        raise AssertionError("Local audit must not use network")

    monkeypatch.setattr(hub, "_request", reject)
    result = hub.check_checkpoint(str(tmp_path))
    assert result["kind"] == "checkpoint_check" and result["status"] == "unknown"
    assert result["provenance"] == []


def test_temporary_view_removed_after_use(network):
    with hub.resolve_hub_snapshot("hf://org/repo", fetch=True) as view:
        root = view.root
        assert root.exists() and (root.stat().st_mode & 0o777) == 0o700
        assert (root / "config.json").is_file()
    assert not root.exists()


def test_declared_unknown_mode_is_not_hidden_by_withheld_stats(network):
    _, files = network
    files["policy_preprocessor.json"] = {"steps": [{"registry_name": "normalizer_processor", "config": {
        "features": {"action": {"type": "ACTION", "shape": [2]}}, "norm_map": {"ACTION": "BOGUS"}},
        "state_file": "state.safetensors"}]}
    files["state.safetensors"] = {}
    result = hub.check_checkpoint("hf://org/repo", fetch=True)
    assert any(f["code"] == "normalization.statistics" and f["status"] == "error"
               and "Unsupported normalization mode" in f["message"] for f in result["findings"])


def test_missing_metadata_is_unknown_not_fault(monkeypatch):
    monkeypatch.setattr(hub, "_request", lambda *a, **k: json.dumps({
        "id": "org/repo", "sha": SHA, "private": False, "siblings": []}).encode())
    result = hub.check_checkpoint("hf://org/repo", fetch=True)
    assert result["status"] == "unknown" and result["counts"]["error"] == 0


def test_alias_identity_is_not_silently_used(monkeypatch):
    monkeypatch.setattr(hub, "_request", lambda *a, **k: json.dumps({
        "id": "new/repo", "sha": SHA, "private": False, "siblings": []}).encode())
    result = hub.check_checkpoint("hf://org/repo", fetch=True)
    assert result["status"] == "unknown"
    assert "aliases are not followed" in result["findings"][0]["message"]


def test_transport_unavailable_is_unknown(monkeypatch):
    def unavailable(*a, **k):
        raise hub.HubUnavailable("Unavailable original metadata")

    monkeypatch.setattr(hub, "_request", unavailable)
    result = hub.check_checkpoint("hf://org/repo", fetch=True)
    assert result["status"] == "unknown" and result["artifacts"] == []
    assert "Unavailable original" in result["findings"][0]["message"]


def test_duplicate_listing_does_not_establish_absence(monkeypatch):
    monkeypatch.setattr(hub, "_request", lambda *a, **k: json.dumps({
        "id": "org/repo", "sha": SHA, "private": False,
        "siblings": [{"rfilename": "config.json"}] * 2}).encode())
    with pytest.raises(InputError, match="duplicate file"):
        hub.check_checkpoint("hf://org/repo", fetch=True)


@pytest.mark.parametrize("payload", [b'{"a":NaN}', b'{"a":Infinity}', b'[]', b'bad json'])
def test_unsafe_json_refused(payload):
    with pytest.raises(InputError):
        hub._json(payload)


def test_download_redirect_cannot_lose_pin(monkeypatch):
    from urllib.error import HTTPError

    class Opener:
        def open(self, request, timeout):
            raise HTTPError(request.full_url, 302, "redirect", {
                "Location": "https://huggingface.co/org/repo/resolve/main/config.json"}, None)

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    with pytest.raises(InputError, match="pinned revision"):
        hub._request(f"https://huggingface.co/org/repo/resolve/{SHA}/config.json", limit=1024, required_sha=SHA)


def test_download_header_commit_drift_refused(monkeypatch):
    class Response:
        status = 200
        headers = {"X-Repo-Commit": "2" * 40}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, n):
            raise AssertionError("Do not read a drifted response")

    class Opener:
        def open(self, request, timeout):
            return Response()

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    with pytest.raises(InputError, match="revision differs"):
        hub._request(f"https://huggingface.co/org/repo/resolve/{SHA}/config.json", limit=1024, required_sha=SHA)


@pytest.mark.parametrize("repo_prefix,filename,cache_type,encoded", [
    ("", "config.json", "models", "config.json"),
    ("datasets/", "meta/info.json", "datasets", "meta%2Finfo.json"),
])
def test_expected_resolve_cache_redirect_keeps_exact_filename(monkeypatch, repo_prefix, filename,
                                                            cache_type, encoded):
    from urllib.error import HTTPError

    calls = []

    class Response:
        status = 200
        headers = {"Content-Length": "2", "X-Repo-Commit": SHA}

        def __enter__(self):
            self.body = iter((b"{}", b""))
            return self

        def __exit__(self, *args):
            pass

        def read(self, n):
            return next(self.body)

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            if len(calls) == 1:
                raise HTTPError(request.full_url, 307, "redirect", {"Location":
                    f"/api/resolve-cache/{cache_type}/org/repo/{SHA}/{encoded}?etag=test"}, None)
            return Response()

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    assert hub._request(f"https://huggingface.co/{repo_prefix}org/repo/resolve/{SHA}/{filename}",
                        limit=1024, required_sha=SHA) == b"{}"
    assert len(calls) == 2


def test_read_failure_cannot_audit_partial_metadata(monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, n):
            raise TimeoutError("generated slow connection")

    class Opener:
        def open(self, request, timeout):
            return Response()

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    result = hub.check_checkpoint("hf://org/repo", fetch=True)
    assert result["status"] == "unknown" and result["artifacts"] == []


def test_transport_does_not_take_ambient_proxy_or_auth(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "https://secret:pw@evil.test")
    monkeypatch.setenv("HF_TOKEN", "private-token")
    opener = hub._opener()
    assert not any(getattr(h, "proxies", {}) for h in opener.handlers)
    assert not any("Auth" in type(h).__name__ or "Cookie" in type(h).__name__ for h in opener.handlers)


def test_private_view_does_not_follow_existing_file_link(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    root = tmp_path / "view"
    root.mkdir()
    (root / "config.json").symlink_to(outside)
    with pytest.raises(OSError):
        hub._write(root, "config.json", b"replacement")
    assert outside.read_text() == "unchanged"


@pytest.mark.parametrize("source", [None, "", 17, False])
def test_required_source_malformed_before_network(source, network):
    calls, _ = network
    with pytest.raises(InputError):
        hub.check_checkpoint(source, fetch=True)
    assert calls == []


@pytest.mark.parametrize("names", [["state"], {"axes": None}])
def test_remote_group_names_keep_dimension_evidence_and_unknown_order(names, network):
    _, files = network
    files["meta/info.json"]["features"]["observation.state"]["names"] = names
    result = hub.check_checkpoint("hf://org/repo", "hf://org/repo", fetch=True)
    assert result["status"] == "error"
    finding = next(f for f in result["findings"] if f["code"] == "dataset.coordinate_names_unresolved")
    assert finding["status"] == "unknown"
    assert finding["evidence"][0]["value"] == names
    assert finding["evidence"][0]["pointer"] == "/features/observation.state/names"
    assert len(finding["evidence"][0]["sha256"]) == 64
    assert len(result["provenance"]) == 2
    assert any(f["code"] == "dataset.feature_shape" and f["status"] == "error"
               for f in result["findings"])


def test_invalid_checkpoint_not_hidden_by_dataset_names_fallback(network):
    _, files = network
    files["config.json"]["input_features"]["observation.state"]["names"] = ["group"]
    files["meta/info.json"]["features"]["observation.state"]["names"] = ["state"]
    with pytest.raises(InputError, match="names length"):
        hub.check_checkpoint("hf://org/repo", "hf://org/repo", fetch=True)


def test_duplicate_names_not_hidden_as_group_format(network):
    _, files = network
    files["meta/info.json"]["features"]["observation.state"]["names"] = ["a", "a", "b"]
    result = hub.check_checkpoint("hf://org/repo", "hf://org/repo", fetch=True)
    codes = {(f["code"], f["status"]) for f in result["findings"]}
    # Reported as repeated labels in their own right, never relabelled as an unresolved group format.
    assert ("dataset.coordinate_names_duplicate", "warning") in codes
    assert not any(code == "dataset.coordinate_names_unresolved" for code, _ in codes)


@pytest.mark.parametrize("expected,malformed", [(["a", "b"], False), (["a", "b", "c"], True)])
def test_custom_reorder_remains_unknown_but_internal_coverage_error_is_retained(expected, malformed, network):
    _, files = network
    files["meta/info.json"]["features"]["action"]["names"] = ["b", "a"]
    files["meta/info.json"]["features"]["observation.state"]["names"] = ["b", "a", "c"]
    files["policy_preprocessor.json"] = {"steps": [
        {"registry_name": "custom_reorder", "config": {}},
        {"registry_name": "relative_actions_processor", "config": {"enabled": True, "action_names": expected}}]}
    files["policy_postprocessor.json"] = {"steps": [
        {"registry_name": "absolute_actions_processor", "config": {"enabled": True}}]}
    result = hub.check_checkpoint("hf://org/repo", "hf://org/repo", fetch=True)
    findings = [f for f in result["findings"] if f["code"] == "action.processor_order_names"]
    if malformed:
        assert any(f["status"] == "error" and "must cover every" in f["message"] for f in findings)
    else:
        assert findings and all(f["status"] == "unknown" for f in findings)


def test_hub_check_never_decides_a_stale_padded_state_declaration(network):
    # Validator round 2: JSON-only intake never has the saved statistics, so for a state-padding policy the
    # width difference stays an undecided warning, and for ACT (above) it stays the baseline error.
    _, files = network
    files["config.json"] = dict(CONFIG, type="smolvla", max_state_dim=32)
    result = hub.check_checkpoint("hf://org/repo", "hf://org/repo", fetch=True)
    shape = next(f for f in result["findings"] if f["code"] == "dataset.feature_shape" and "observation.state" in f["message"])
    assert shape["status"] == "warning" and "never decides" in shape["message"]
    assert "modeling_smolvla.py:370-371" in shape["message"]
