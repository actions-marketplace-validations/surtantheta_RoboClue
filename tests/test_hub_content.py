"""Opt-in public dataset content intake: hash-verified Parquet rows, never videos, never unverified bytes."""
import email.message
import hashlib
import io
import json
from urllib.error import HTTPError

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor import hub
from policy_doctor.artifacts import InputError
from policy_doctor.cli import main
from policy_doctor.windows import audit_episode_windows

SHA = "b" * 40


def _bytes(table):
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(table), sink)
    return sink.getvalue()


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A fake public dataset whose declared totals contradict its rows (as in SO-101-robot-learning/so101_pick_place)."""
    rows = [{"index": i, "episode_index": i // 10, "frame_index": i % 10, "action": [float(i), 0.0]} for i in range(20)]
    meta = [{"episode_index": e, "length": 10, "dataset_from_index": e * 10, "dataset_to_index": (e + 1) * 10,
             "data/chunk_index": 0, "data/file_index": 0} for e in range(2)]
    info = {"codebase_version": "v3.0", "total_frames": 0, "total_episodes": 0, "fps": 10,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": {"action": {"dtype": "float32", "shape": [2]}}}
    files = {"meta/info.json": json.dumps(info).encode(), "meta/episodes/chunk-000/file-000.parquet": _bytes(meta),
             "data/chunk-000/file-000.parquet": _bytes(rows), "videos/chunk-000/file-000.mp4": b"never fetched"}
    local = tmp_path / "local"
    for name, data in files.items():
        (local / name).parent.mkdir(parents=True, exist_ok=True)
        (local / name).write_bytes(data)
    state = {"files": files, "calls": [], "redirect_host": "us.aws.cdn.hf.co", "corrupt": False, "extra_siblings": [],
             "served": {}}

    def listing():
        siblings = []
        for name, data in state["files"].items():
            item = {"rfilename": name, "size": len(data), "blobId": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
            if not name.endswith(".json"):
                item["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            siblings.append(item)
        siblings += state["extra_siblings"]
        return {"id": "org/data", "sha": SHA, "private": False, "siblings": siblings}

    def request(url, *, limit, required_sha=None):
        state["calls"].append(url)
        if "/api/datasets/" in url:
            return json.dumps(listing()).encode()
        name = url.split(f"/resolve/{SHA}/", 1)[1]
        return state["served"].get(name, state["files"][name])

    class Response:
        def __init__(self, data):
            self.stream, self.headers = io.BytesIO(data), {}

        def read(self, n=-1):
            return self.stream.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Opener:
        def open(self, request, timeout=None):
            url = request.full_url
            state["calls"].append(url)
            if url.startswith("https://huggingface.co/"):
                name = url.split(f"/resolve/{SHA}/", 1)[1]
                headers = email.message.Message()
                headers["Location"] = f"https://{state['redirect_host']}/xet/{name}"
                headers["X-Linked-Etag"] = '"' + hashlib.sha256(state["files"][name]).hexdigest() + '"'
                raise HTTPError(url, 302, "Found", headers, None)
            name = url.split("/xet/", 1)[1]
            data = state["files"][name]
            return Response(data + b"x" if state["corrupt"] else data)

    monkeypatch.setattr(hub, "_request", request)
    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    return state, local


def test_fetched_dataset_matches_the_local_audit_and_never_reads_videos(remote, capsys):
    state, local = remote
    assert main(["audit-windows", "hf://org/data", "--fetch", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    expected = audit_episode_windows(str(local))
    assert report["status"] == expected["status"] == "error"
    assert [f["code"] for f in report["findings"]] == [f["code"] for f in expected["findings"]]
    assert report["provenance"]["resolved_revision"] == SHA
    assert {f["path"] for f in report["provenance"]["files"]} == {
        "meta/info.json", "meta/episodes/chunk-000/file-000.parquet", "data/chunk-000/file-000.parquet"}
    assert not any("videos/" in url for url in state["calls"])
    assert all(e["path"].startswith(f"hf://org/data@{SHA}/") for f in report["findings"] for e in f.get("evidence", [])
               if "path" in e)


def test_no_network_without_fetch(remote, capsys):
    state, _ = remote
    assert main(["audit-windows", "hf://org/data"]) == 2
    assert state["calls"] == [] and "fetch" in capsys.readouterr().err.lower()


def test_redirect_away_from_hugging_face_is_refused(remote):
    state, _ = remote
    state["redirect_host"] = "evil.example.com"
    with pytest.raises(InputError, match="left HTTPS Hugging Face hosts"):
        with hub.resolve_dataset_content("hf://org/data", fetch=True):
            pass


def test_bytes_that_do_not_match_the_pinned_sha256_are_refused(remote):
    state, _ = remote
    state["corrupt"] = True
    with pytest.raises(InputError, match="listed size|SHA-256"):
        with hub.resolve_dataset_content("hf://org/data", fetch=True):
            pass


def test_download_cap_is_enforced_before_any_content_request(remote):
    state, _ = remote
    with pytest.raises(InputError, match="above the 10-byte limit"):
        with hub.resolve_dataset_content("hf://org/data", fetch=True, max_bytes=10):
            pass
    assert all("/api/datasets/" in url for url in state["calls"])


def test_parquet_without_a_listed_content_hash_is_refused(remote, monkeypatch):
    state, _ = remote
    original = hub._request

    def without_hash(url, *, limit, required_sha=None):
        data = original(url, limit=limit, required_sha=required_sha)
        if "/api/datasets/" in url:
            listing = json.loads(data)
            for item in listing["siblings"]:
                item.pop("lfs", None)
            return json.dumps(listing).encode()
        return data

    monkeypatch.setattr(hub, "_request", without_hash)
    with pytest.raises(InputError, match="content SHA-256"):
        with hub.resolve_dataset_content("hf://org/data", fetch=True):
            pass


# ---- Validator round 2 (d5fdd9c): fetch-path gaps ----

def test_json_that_does_not_match_the_listed_size_and_blob_id_is_refused(remote):
    # Validator R3: 300 bytes were accepted for a file listed at 5,000.
    state, _ = remote
    state["served"]["meta/info.json"] = state["files"]["meta/info.json"][:-1] + b" "
    with pytest.raises(InputError, match="git blob ID"):
        with hub.resolve_dataset_content("hf://org/data", fetch=True):
            pass


def test_unusual_file_names_exit_2_before_any_download(remote, capsys):
    # Validator: spaces, non-ASCII and backslashes crashed with exit 1 (read as "error found").
    state, _ = remote
    for name in ("data/a b/c.parquet", "data/a/\u0444.parquet", "data/..\\..\\evil/x.parquet"):
        state["calls"].clear()
        state["extra_siblings"] = [{"rfilename": name, "size": 4, "lfs": {"sha256": "0" * 64}}]
        assert main(["audit-windows", "hf://org/data", "--fetch", "--json"]) == 2, name
        assert "unsupported name" in capsys.readouterr().err
        assert all("/api/datasets/" in url for url in state["calls"])


def test_duplicate_listing_entries_and_hashless_lfs_entries_exit_2(remote, capsys):
    state, _ = remote
    data = state["files"]["meta/info.json"]
    state["extra_siblings"] = [{"rfilename": "meta/info.json", "size": len(data)}]
    assert main(["audit-windows", "hf://org/data", "--fetch"]) == 2
    assert "repeats" in capsys.readouterr().err
    state["extra_siblings"] = [{"rfilename": "meta/stats.json", "size": 2, "lfs": {"size": 2}}]
    assert main(["audit-windows", "hf://org/data", "--fetch"]) == 2
    assert "content SHA-256" in capsys.readouterr().err


def test_file_count_is_capped_before_any_content_request(remote, monkeypatch):
    state, _ = remote
    monkeypatch.setattr(hub, "MAX_CONTENT_FILES", 2)
    with pytest.raises(InputError, match="file limit"):
        with hub.resolve_dataset_content("hf://org/data", fetch=True):
            pass
    assert all("/api/datasets/" in url for url in state["calls"])


def test_report_provenance_is_bounded_and_pinned_by_digest(remote, monkeypatch, capsys):
    # Validator R9: 3,003 files produced an 870 KB report because provenance was attached after the size check.
    monkeypatch.setattr(hub, "MAX_PROVENANCE_FILES", 1)
    assert main(["audit-windows", "hf://org/data", "--fetch", "--json"]) == 1
    provenance = json.loads(capsys.readouterr().out)["provenance"]
    assert provenance["file_count"] == 3 and provenance["files_truncated"] and len(provenance["files"]) == 1
    assert len(provenance["files_manifest_sha256"]) == 64


def test_cdn_redirect_with_a_nonstandard_port_is_refused(remote):
    state, _ = remote
    state["redirect_host"] = "us.aws.cdn.hf.co:8443"
    with pytest.raises(InputError, match="left HTTPS Hugging Face hosts"):
        with hub.resolve_dataset_content("hf://org/data", fetch=True):
            pass


def test_invalid_urls_in_low_level_requests_are_input_errors(monkeypatch, tmp_path):
    # Validator R11: the low-level functions raised http.client.InvalidURL / UnicodeEncodeError (exit 1).
    import http.client

    class Opener:
        def open(self, request, timeout=None):
            raise http.client.InvalidURL("URL can't contain control characters")

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    with pytest.raises(InputError, match="not valid"):
        hub._request("https://huggingface.co/api/datasets/org/data", limit=10)
    with pytest.raises(InputError, match="not valid"):
        hub._download_blob("https://huggingface.co/datasets/org/data/resolve/x/y", tmp_path / "y", size=1, sha256="0" * 64)


def test_truncated_http_responses_are_unavailable_not_tracebacks(monkeypatch):
    import http.client

    class Response:
        headers, status = {}, 200

        def read1(self, n):
            raise http.client.IncompleteRead(b"partial", 10)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Opener:
        def open(self, request, timeout=None):
            return Response()

    monkeypatch.setattr(hub, "_opener", lambda: Opener())
    with pytest.raises(hub.HubUnavailable):
        hub._request("https://huggingface.co/api/datasets/org/data", limit=100)


def test_case_only_duplicate_names_are_repeats(remote):
    # Validator round 3 (8d): FILE-000 beside file-000 collides on case-insensitive filesystems.
    state, _ = remote
    data = state["files"]["data/chunk-000/file-000.parquet"]
    state["extra_siblings"] = [{"rfilename": "data/chunk-000/FILE-000.parquet", "size": len(data),
                                "lfs": {"sha256": hashlib.sha256(data).hexdigest()}}]
    with pytest.raises(InputError, match="repeats"):
        with hub.resolve_dataset_content("hf://org/data", fetch=True):
            pass


def test_errors_after_download_name_the_hub_identity_not_the_temp_path(remote, monkeypatch, capsys):
    # Validator round 3 (8b): a hostile LFS meta/info.json exited 2 but printed the private temporary path.
    import policy_doctor.windows as windows

    def failing(path, **kwargs):
        raise InputError(f"Cannot parse {path}/meta/info.json")

    monkeypatch.setattr(windows, "audit_episode_windows", failing)
    assert main(["audit-windows", "hf://org/data", "--fetch"]) == 2
    err = capsys.readouterr().err
    assert "policy-doctor-hub-data-" not in err and f"hf://org/data@{SHA}/meta/info.json" in err


def test_fetched_window_report_is_bounded_after_the_hub_path_rewrite(tmp_path):
    # Validator round 4: ~800 shards passed the local 192 KiB bound, then exceeded 224 KiB once every path became
    # hf://ORG/REPO@<sha>/..., and the CLI exited 2. The artifact list is bounded again after the rewrite.
    root = tmp_path / "view"
    artifacts = [{"path": str(root / f"data/chunk-000/file-{i:03d}.parquet"), "sha256": "0" * 64, "size_bytes": 1,
                  "role": "dataset_parquet"} for i in range(1000)]
    report = {"kind": "episode_window_audit", "status": "pass", "findings": [], "artifacts": artifacts}
    view = hub.HubSnapshot(root, {"source": "hf://unitreerobotics/G1_WBT_Dex1_Put_Clothes_into_Washing_Machine",
                                  "resolved_revision": SHA, "files": []}, {})
    out = hub.portable_dataset_report(report, view)
    assert out["artifacts_truncated"] and out["artifact_count"] == 1000
    assert len(json.dumps(out).encode()) <= 224 * 1024
