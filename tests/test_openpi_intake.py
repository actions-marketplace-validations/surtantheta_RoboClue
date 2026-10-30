"""Original captured public bodies and hostile controls
no live HTTP."""
import base64
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError

import pytest

from policy_doctor import openpi_intake as intake
from policy_doctor.artifacts import InputError
from policy_doctor.readiness_intake import ReadinessIntakeBudget

FIXTURE = Path(__file__).parent / "fixtures/openpi_original_20261024"
SOURCE = "gs://openpi-assets/checkpoints/pi0_base/assets/droid/norm_stats.json"
GENERATION = "1748456810418896"
META_URL = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o/checkpoints%2Fpi0_base%2Fassets%2Fdroid%2Fnorm_stats.json"
BODY_URL = "https://storage.googleapis.com/openpi-assets/checkpoints/pi0_base/assets/droid/norm_stats.json?generation=" + GENERATION


class Response:
    def __init__(self, raw, url, *, headers=None, status=200, fail_after=None):
        self.raw, self.url, self.offset = raw, url, 0
        self.status = status
        self.headers = {"Content-Length": str(len(raw)), **(headers or {})}
        self.fail_after = fail_after
        self.reads = 0
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def geturl(self):
        return self.url
    def read(self, size):
        self.reads += 1
        if self.fail_after is not None and self.offset >= self.fail_after:
            raise OSError("generated partial transport")
        value = self.raw[self.offset:self.offset + size]
        if self.fail_after is not None:
            value = value[:max(0, self.fail_after - self.offset)]
        self.offset += len(value)
        return value
    read1 = read


def delivered(monkeypatch, *, body=None, metadata=None, body_headers=None, meta_headers=None,
              body_status=200, body_url=None, fail_after=None, budget=None, generation=None):
    body = (FIXTURE / "norm_stats.json").read_bytes() if body is None else body
    meta_raw = (FIXTURE / "object_metadata.json").read_bytes() if metadata is None else json.dumps(metadata).encode()
    meta_url = META_URL if generation is None else META_URL + "?generation=" + generation
    calls = []
    starting_requests = budget.logical_requests if budget is not None else 0
    responses = [Response(meta_raw, meta_url, headers=meta_headers),
                 Response(body, body_url or BODY_URL, status=body_status, fail_after=fail_after,
                          headers={"x-goog-generation": GENERATION, **(body_headers or {})})]
    class Opener:
        def open(self, request, timeout):
            assert request.full_url in (meta_url, BODY_URL)
            assert timeout == 15
            assert request.get_header("Authorization") is None
            if budget is not None:
                assert budget.used_bytes > 0 and budget.logical_requests == starting_requests + len(calls) + 1
            response = responses[len(calls)]
            calls.append(request.full_url)
            return response
    monkeypatch.setattr(intake, "_opener", Opener)
    return calls, responses


def for_body(body):
    metadata = json.loads((FIXTURE / "object_metadata.json").read_bytes())
    metadata.update(size=str(len(body)), md5Hash=base64.b64encode(hashlib.md5(body).digest()).decode())
    return metadata


def test_original_generation_identity_null_quantiles_and_shared_budget(monkeypatch):
    raw = (FIXTURE / "norm_stats.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "c6f9a4794b1df4901e618db3d3b5e02126dcc7c8d95b8f9735b5bc9ed75b95c9"
    budget = ReadinessIntakeBudget(max_bytes=2 * 1024 ** 2, max_files=8)
    calls, _ = delivered(monkeypatch, budget=budget)
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True, budget=budget)
    assert result.status == "pass" and result.raw == raw
    assert result.metadata_raw == (FIXTURE / "object_metadata.json").read_bytes()
    assert result.document == json.loads(raw) and result.document["norm_stats"]["state"]["q01"] is None
    assert result.provenance["generation"] == GENERATION and result.provenance["content_identity_verified"]
    assert result.provenance["historical_training_authority"] == "not_authenticated"
    assert result.observations["roles"]["actions"]["field_shapes"]["mean"] == [32]
    assert calls == [META_URL, BODY_URL]
    assert budget.logical_requests == 2 and budget.used_bytes == len(result.metadata_raw) + len(raw)
    assert budget.used_bytes == budget.successful_body_bytes
    assert "wire" not in result.accounting["scope"].lower()


@pytest.mark.parametrize("source", [
    SOURCE.replace("openpi-assets", "evil"), SOURCE + "?generation=1", SOURCE + "#fragment",
    SOURCE.replace("gs://", "https://"), SOURCE.replace("assets/droid", "assets/../droid"),
    SOURCE.replace("assets/droid", "assets/%2fdroid"), SOURCE.replace("norm_stats.json", "params.safetensors"),
    "gs://user@openpi-assets/checkpoints/x/assets/droid/norm_stats.json",
    "gs://openpi-assets:443/checkpoints/x/assets/droid/norm_stats.json",
])
def test_unapproved_authority_or_path_zero_http(monkeypatch, source):
    monkeypatch.setattr(intake, "_opener", lambda: pytest.fail("no HTTP before source authorization"))
    with pytest.raises(InputError):
        intake.resolve_openpi_statistics(source, fetch=True)


def test_default_permission_and_strict_arguments_zero_http(monkeypatch):
    monkeypatch.setattr(intake, "_opener", lambda: pytest.fail("no HTTP before authorization"))
    for options in [{}, {"fetch": 1}, {"fetch": "true"}, {"fetch": True, "read_roots": "/"}]:
        with pytest.raises(InputError):
            intake.resolve_openpi_statistics(SOURCE, **options)


@pytest.mark.parametrize("change", [{"bucket": "elsewhere"}, {"name": "checkpoints/other/assets/droid/norm_stats.json"},
    {"generation": "0"}, {"generation": 1}, {"size": "1048577"}, {"size": "-1"},
    {"size": True}, {"md5Hash": "bad"}, {"md5Hash": base64.b64encode(b"short").decode()}])
def test_incomplete_or_wrong_selected_metadata_unknown(monkeypatch, change):
    metadata = json.loads((FIXTURE / "object_metadata.json").read_bytes())
    metadata.update(change)
    calls, _ = delivered(monkeypatch, metadata=metadata)
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True)
    assert result.status == "unknown" and result.document is None and result.metadata == metadata
    assert len(calls) == 1 and result.metadata_raw is not None


@pytest.mark.parametrize("options", [
    {"body_headers": {"x-goog-generation": "2"}}, {"body_headers": {"Content-Length": "9999"}},
    {"body_headers": {"Content-Length": "4"}}, {"body_headers": {"Content-Encoding": "gzip"}},
    {"body_status": 206}, {"body_url": "https://evil.example/stats"},
    {"metadata": dict(json.loads((FIXTURE / "object_metadata.json").read_bytes()), md5Hash=base64.b64encode(b"0" * 16).decode())},
])
def test_generation_length_encoding_md5_and_complete_response_guards(monkeypatch, options):
    calls, _ = delivered(monkeypatch, **options)
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True)
    assert result.status == "unknown" and result.document is None and len(calls) == 2


def test_partial_body_unknown_preserves_full_reservation(monkeypatch):
    budget = ReadinessIntakeBudget()
    calls, _ = delivered(monkeypatch, fail_after=7, budget=budget)
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True, budget=budget)
    assert result.status == "unknown" and result.document is None and result.raw is None
    assert budget.used_bytes == len(result.metadata_raw) + 2414 and budget.logical_requests == 2
    assert budget.successful_body_bytes == len(result.metadata_raw) and len(calls) == 2


def test_redirect_refused_without_following_or_media_link(monkeypatch):
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            raise HTTPError(request.full_url, 302, "generated", {"Location": "https://evil.example"}, None)
    monkeypatch.setattr(intake, "_opener", Opener)
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True)
    assert result.status == "unknown" and len(calls) == 1
    assert result.accounting["response_body_bytes_or_failed_reservations"] == 65537
    assert intake._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example") is None


def test_impossible_budget_and_exhausted_requests_no_http(monkeypatch):
    monkeypatch.setattr(intake, "_opener", lambda: pytest.fail("no HTTP without reservation"))
    budget = ReadinessIntakeBudget(max_bytes=1, max_files=8)
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True, budget=budget)
    assert result.status == "unknown" and budget.logical_requests == budget.used_bytes == 0
    budget = ReadinessIntakeBudget(max_files=1)
    budget.logical_requests = 1
    assert intake.resolve_openpi_statistics(SOURCE, fetch=True, budget=budget).status == "unknown"


def test_metadata_succeeds_but_declared_object_cannot_fit_no_body_call(monkeypatch):
    budget = ReadinessIntakeBudget(max_bytes=2000)
    calls, _ = delivered(monkeypatch)
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True, budget=budget)
    assert result.status == "unknown" and len(calls) == 1 and result.metadata is not None


@pytest.mark.parametrize("body", [b"\xff", b'{"norm_stats":' + b"[" * 33 + b"0" + b"]" * 33 + b"}",
    b'{"norm_stats":{},"norm_stats":{}}', b'{"norm_stats":{"state":{"mean":[NaN],"std":[1]}}}',
    b'{"norm_stats":{"state":{"mean":[1e999],"std":[1]}}}', b"[]"])
def test_unsupported_json_unknown_preserves_original_verified_bytes(monkeypatch, body):
    delivered(monkeypatch, body=body, metadata=for_body(body))
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True)
    assert result.status == "unknown" and result.raw == body and result.document is None
    assert result.provenance["content_identity_verified"]


@pytest.mark.parametrize("document", [
    {"norm_stats": {"native_extra_role": {"mean": [1, 2], "std": [1, 1], "q01": None, "q99": None}}},
    {"norm_stats": {"state": {"mean": [1, 2], "std": [1, 1], "count": 4}}},
    {"norm_stats": {"state": {"mean": [[1], [2, 3]], "std": [1, 1]}}},
    {"norm_stats": {"state": {"mean": [True], "std": [1]}}},
    {"norm_stats": {"state": {"mean": [1], "std": [1]}} , "native_extension": {"anything": "preserved"}},
])
def test_native_extras_roles_and_unsupported_shapes_preserved(monkeypatch, document):
    body = json.dumps(document).encode()
    delivered(monkeypatch, body=body, metadata=for_body(body))
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True)
    assert result.document == document and result.raw == body
    assert all("corrupt" not in x for x in result.reasons)
    if "native_extra_role" in document["norm_stats"]:
        assert result.status == "pass" and "native_extra_role" in result.observations["roles"]
    else:
        assert result.status == "unknown"


def test_local_original_no_network_and_no_budget_network_charge(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "_opener", lambda: pytest.fail("local read has no HTTP"))
    path = tmp_path / "norm_stats.json"
    path.write_bytes((FIXTURE / "norm_stats.json").read_bytes())
    result = intake.resolve_openpi_statistics(str(path), read_roots=[tmp_path])
    assert result.status == "pass" and result.document == json.loads(path.read_bytes())
    assert result.accounting["logical_requests"] == 0 and result.accounting["response_body_bytes_or_failed_reservations"] == 0


def test_outside_read_root_before_any_filesystem_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(intake.os, "open", lambda *a, **k: pytest.fail("authorization before filesystem probe"))
    with pytest.raises(InputError, match="outside"):
        intake.resolve_openpi_statistics("/etc/norm_stats.json", read_roots=[tmp_path])


def test_local_symlink_ancestor_fifo_directory_and_oversize_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    path = real / "norm_stats.json"
    path.write_bytes((FIXTURE / "norm_stats.json").read_bytes())
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    (real / "file-link.json").symlink_to(path)
    fifo = real / "fifo"
    os.mkfifo(fifo)
    big = real / "big"
    big.write_bytes(b"x" * (1024 * 1024 + 1))
    for candidate in [tmp_path / "link/norm_stats.json", real / "file-link.json", fifo, real, big]:
        with pytest.raises(InputError):
            intake.resolve_openpi_statistics(str(candidate), read_roots=[tmp_path])


def test_oversize_stream_without_length_rejected_at_probe(monkeypatch):
    body = (FIXTURE / "norm_stats.json").read_bytes() + b"x"
    calls, responses = delivered(monkeypatch, body=body)
    responses[1].headers.pop("Content-Length")
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True)
    assert result.status == "unknown" and result.raw is None and len(calls) == 2
    assert responses[1].offset == 2414


def test_node_bound_unknown_on_valid_bounded_native_json(monkeypatch):
    body = json.dumps({"norm_stats": {"state": {"mean": [0] * 100001, "std": [1]}}}).encode()
    delivered(monkeypatch, body=body, metadata=for_body(body))
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True)
    assert result.status == "unknown" and result.raw == body and result.document is None
    assert any("value-count" in reason for reason in result.reasons)


def test_source_and_authorized_root_bounds_before_transport(monkeypatch):
    monkeypatch.setattr(intake, "_opener", lambda: pytest.fail("oversized source/root must not reach HTTP"))
    for source, options in [(SOURCE + "x" * 4096, {"fetch": True}),
                            (SOURCE, {"fetch": True, "read_roots": ["/" + "x" * 4096]}),
                            ("gs://[invalid", {"fetch": True}),
                            (SOURCE, {"fetch": True, "read_roots": ["/\x00"]})]:
        with pytest.raises(InputError):
            intake.resolve_openpi_statistics(source, **options)


def test_repeated_intakes_share_existing_ledger_without_reset(monkeypatch):
    budget = ReadinessIntakeBudget(max_files=4)
    for count in [2, 4]:
        delivered(monkeypatch, budget=budget)
        result = intake.resolve_openpi_statistics(SOURCE, fetch=True, budget=budget)
        assert result.status == "pass" and budget.logical_requests == count
    assert budget.used_bytes == 2 * (len(result.raw) + len(result.metadata_raw))
    monkeypatch.setattr(intake, "_opener", lambda: pytest.fail("shared request bound must refuse"))
    refused = intake.resolve_openpi_statistics(SOURCE, fetch=True, budget=budget)
    assert refused.status == "unknown" and budget.logical_requests == 4


@pytest.mark.parametrize("parent_in", ["source", "read_root"])
def test_parent_components_refused_before_probes(tmp_path, monkeypatch, parent_in):
    source = tmp_path / "link/../norm_stats.json" if parent_in == "source" else tmp_path / "norm_stats.json"
    roots = [tmp_path / "link/.."] if parent_in == "read_root" else [tmp_path]
    monkeypatch.setattr(intake.os, "open", lambda *a, **k: pytest.fail("literal parent traversal must not probe a rewritten file"))
    with pytest.raises(InputError):
        intake.resolve_openpi_statistics(str(source), read_roots=roots)


@pytest.mark.parametrize("generation", [True, 1, "0", "01", "-1", "+1", " 1", "1 ", "1?x", "1" * 31, "１２"])
def test_generation_argument_strict_no_http(monkeypatch, generation):
    monkeypatch.setattr(intake, "_opener", lambda: pytest.fail("invalid generation must not reach HTTP"))
    with pytest.raises(InputError, match="generation"):
        intake.resolve_openpi_statistics(SOURCE, fetch=True, generation=generation)


def test_requested_generation_metadata_and_body_exact_replay(monkeypatch):
    budget = ReadinessIntakeBudget(max_files=4)
    delivered(monkeypatch, budget=budget)
    initial = intake.resolve_openpi_statistics(SOURCE, fetch=True, budget=budget)
    calls, _ = delivered(monkeypatch, budget=budget, generation=initial.provenance["generation"])
    replay = intake.resolve_openpi_statistics(SOURCE, fetch=True, budget=budget, generation=initial.provenance["generation"])
    assert initial.status == replay.status == "pass" and replay.raw == initial.raw
    assert calls == [META_URL + "?generation=" + GENERATION, BODY_URL]
    assert replay.provenance["generation"] == replay.provenance["requested_generation"] == GENERATION
    assert budget.logical_requests == 4


def test_wrong_requested_generation_unknown_before_body_request(monkeypatch):
    calls, _ = delivered(monkeypatch, generation="2")
    result = intake.resolve_openpi_statistics(SOURCE, fetch=True, generation="2")
    assert result.status == "unknown" and result.document is None and result.raw is None
    assert len(calls) == 1 and calls[0].endswith("?generation=2")
    assert any("explicitly requested" in reason for reason in result.reasons)


def test_local_generation_refused_before_any_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(intake.os, "open", lambda *a, **k: pytest.fail("local generation must refuse before probing"))
    with pytest.raises(InputError, match="local"):
        intake.resolve_openpi_statistics(str(tmp_path / "norm_stats.json"), generation=GENERATION)
