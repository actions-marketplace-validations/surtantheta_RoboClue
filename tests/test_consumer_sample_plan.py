"""Generated metadata-only planning; no real dataset or native execution."""

import copy
import hashlib
import json

import pytest
from test_consumer_numeric import artifacts

from policy_doctor import consumer_sample_plan as planner
from policy_doctor.artifacts import InputError
from policy_doctor.consumer_context import resolve_native_processor_context


@pytest.fixture
def job(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config, stats = artifacts()
    (checkpoint / "processor_config.json").write_text(json.dumps(config))
    (checkpoint / "statistics.json").write_text(json.dumps(stats))
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    return {"dataset": dataset, "checkpoint": checkpoint, "root": tmp_path}


def metadata(job, episodes):
    info = {"codebase_version": "v2.1", "data_path": planner.DATA_TEMPLATE,
            "chunks_size": 1000, "features": {}, "total_episodes": len(episodes)}
    (job["dataset"] / "meta/info.json").write_text(json.dumps(info))
    (job["dataset"] / "meta/episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": episode, "length": length}) + "\n"
        for episode, length in episodes))


def context(job):
    return resolve_native_processor_context(job["checkpoint"], read_roots=[job["root"]])


def run(job, ctx=None):
    return planner.plan_consumer_samples(job["dataset"], ctx or context(job),
                                         read_roots=[job["root"]])


def test_nine_windows_follow_eligible_rank_and_stored_id_not_metadata_position(job):
    metadata(job, [(90, 49), (72, 154), (3, 55), (250, 171), (8, 60), (41, 189), (0, 40)])
    report = run(job)
    assert report["status"] == report["planning_status"] == "PASS"
    assert [x["episode_index"] for x in report["selected_episodes"]] == [72, 3, 41]
    assert [x["metadata_position"] for x in report["selected_episodes"]] == [1, 2, 5]
    assert [x["anchors"] for x in report["selected_episodes"]] == [[0, 52, 104], [0, 2, 5], [0, 69, 139]]
    assert report["counts"] == {"declared_episodes_total": 7, "episodes_total": 7, "eligible_episodes": 5,
        "short_episodes": 2, "selected_episodes": 3, "selected_windows": 9}
    assert all(x["allow_padding"] is False for x in report["samples"])
    assert all(value == "UNKNOWN" for value in report["boundaries"].values())
    for artifact in report["readset"]:
        data = (job["dataset"] / artifact["path"]).read_bytes()
        assert artifact["size_bytes"] == len(data)
        assert artifact["sha256"] == hashlib.sha256(data).hexdigest()
    assert report["coverage"]["dataset_arrays_read"] is False
    assert "match these dataset metadata hashes" in report["integration_requirement"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("episodes,expected", [([(99, 50)], [(99, 0)]),
    ([(99, 51)], [(99, 0), (99, 1)]),
    ([(99, 50), (4, 52)], [(99, 0), (4, 0), (4, 1), (4, 2)])])
def test_duplicate_ranks_and_anchors_removed_without_padding(job, episodes, expected):
    metadata(job, episodes)
    assert [(x["episode_index"], x["anchor_index"]) for x in run(job)["samples"]] == expected


def test_no_eligible_metadata_returns_no_invented_samples(job):
    metadata(job, [(72, 49), (250, 1)])
    report = run(job)
    assert report["status"] == report["planning_status"] == "UNKNOWN"
    assert report["samples"] == []
    assert report["counts"]["eligible_episodes"] == 0
    assert report["findings"][0]["code"] == "consumer.plan_no_eligible_episode"


def test_capacity_error_retained_independently_of_usable_metadata(job):
    metadata(job, [(72, 50)])
    path = job["checkpoint"] / "processor_config.json"
    config = json.loads(path.read_text())
    config["processor_kwargs"]["max_action_dim"] = 1
    path.write_text(json.dumps(config))
    report = run(job)
    assert report["planning_status"] == "PASS" and report["status"] == "ERROR"
    assert report["context_status"] == "ERROR"
    assert report["findings"][0]["cause_scope"] == "saved_native_context"
    assert report["samples"] == [{"episode_index": 72, "anchor_index": 0, "allow_padding": False}]


def test_caller_edited_status_cannot_hide_fresh_capacity_error(job):
    metadata(job, [(72, 50)])
    path = job["checkpoint"] / "processor_config.json"
    config = json.loads(path.read_text())
    config["processor_kwargs"]["max_action_dim"] = 1
    path.write_text(json.dumps(config))
    ctx = context(job)
    ctx["status"], ctx["findings"], ctx["axes"] = "PASS", [], {}
    assert run(job, ctx)["status"] == "ERROR"


def test_saved_context_drift_is_not_silently_refreshed(job):
    metadata(job, [(72, 50)])
    ctx = context(job)
    path = job["checkpoint"] / "statistics.json"
    path.write_text(path.read_text() + "\n")
    report = run(job, ctx)
    assert report["status"] == "UNKNOWN" and report["samples"] == []
    assert report["readset"] == []


@pytest.mark.parametrize("change", ["v3", "bad_template", "bool_chunk", "features"])
def test_unsupported_info_never_creates_partial_plan(job, change):
    metadata(job, [(72, 50)])
    path = job["dataset"] / "meta/info.json"
    info = json.loads(path.read_text())
    if change == "v3":
        info["codebase_version"] = "v3.0"
    elif change == "bad_template":
        info["data_path"] = "../outside"
    elif change == "bool_chunk":
        info["chunks_size"] = True
    else:
        info["features"] = []
    path.write_text(json.dumps(info))
    assert run(job)["planning_status"] == "UNKNOWN"
    assert run(job)["samples"] == []


@pytest.mark.parametrize("body", ["", "[]\n", "{\"episode_index\":72,\"length\":50,\"length\":51}\n",
    '{"episode_index":72,"length":NaN}\n', '{"episode_index":true,"length":50}\n',
    '{"episode_index":72,"length":0}\n', '{"episode_index":72,"length":4097}\n',
    '{"episode_index":72,"length":50}\n{"episode_index":72,"length":51}\n'])
def test_malformed_episode_metadata_is_unknown(job, body):
    metadata(job, [(72, 50)])
    (job["dataset"] / "meta/episodes.jsonl").write_text(body)
    report = run(job)
    assert report["planning_status"] == "UNKNOWN" and report["samples"] == []


@pytest.mark.parametrize("target", ["meta", "info.json", "episodes.jsonl"])
def test_symlinks_refused_even_inside_authorized_root(job, target):
    metadata(job, [(72, 50)])
    path = job["dataset"] / ("meta" if target == "meta" else "meta/" + target)
    moved = path.with_name(path.name + ".real")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=target == "meta")
    report = run(job)
    assert report["planning_status"] == "UNKNOWN" and report["samples"] == []


def test_metadata_change_between_first_and_stability_reads_is_unknown(job, monkeypatch):
    metadata(job, [(72, 50)])
    original = planner._read_metadata
    calls = []
    def change(fd, relative):
        calls.append(relative)
        if len(calls) == 3:
            path = job["dataset"] / "meta/info.json"
            path.write_text(path.read_text() + "\n")
        return original(fd, relative)
    monkeypatch.setattr(planner, "_read_metadata", change)
    report = run(job)
    assert report["samples"] == []
    assert report["planning_status"] == "UNKNOWN"


def test_replaced_root_refused_even_though_metadata_reads_use_retained_fd(job, monkeypatch):
    metadata(job, [(72, 50)])
    original = planner._read_metadata
    calls = []
    def replace(fd, relative):
        calls.append(relative)
        value = original(fd, relative)
        if len(calls) == 2:
            job["dataset"].rename(job["dataset"].with_name("moved_dataset"))
            job["dataset"].mkdir()
        return value
    monkeypatch.setattr(planner, "_read_metadata", replace)
    report = run(job)
    assert report["planning_status"] == "UNKNOWN" and report["samples"] == []


def test_all_literal_paths_authorized_before_helper_reads(job, monkeypatch):
    ctx = context(job)
    monkeypatch.setattr(planner, "load_consumer_numeric_parameters", lambda *a, **k: pytest.fail("read"))
    for invalid in (job["root"].parent / "outside", "~/dataset", "/tmp/x/../dataset"):
        with pytest.raises(InputError):
            planner.plan_consumer_samples(invalid, ctx, read_roots=[job["root"]])
    invalid = copy.deepcopy(ctx)
    invalid["checkpoint_directory"] = str(job["root"].parent / "outside")
    with pytest.raises(InputError):
        run(job, invalid)


def test_non_50_offsets_context_does_not_plan(job):
    metadata(job, [(72, 50)])
    path = job["checkpoint"] / "processor_config.json"
    config = json.loads(path.read_text())
    for modalities in config["processor_kwargs"]["modality_configs"].values():
        modalities["action"]["delta_indices"] = list(range(49))
    path.write_text(json.dumps(config))
    report = run(job)
    assert report["planning_status"] == "UNKNOWN" and report["samples"] == []


def test_byte_bound_and_episode_count_bound(job):
    metadata(job, [(72, 50)])
    path = job["dataset"] / "meta/episodes.jsonl"
    path.write_bytes(b" " * (planner.MAX_METADATA + 1))
    assert run(job)["planning_status"] == "UNKNOWN"
    path.write_text("".join(json.dumps({"episode_index": i, "length": 50}) + "\n"
                            for i in range(planner.MAX_EPISODES + 1)))
    assert run(job)["planning_status"] == "UNKNOWN"


def test_declared_count_mismatch_does_not_hide_selected_native_windows(job):
    metadata(job, [(72, 154), (250, 171)])
    path = job["dataset"] / "meta/info.json"
    info = json.loads(path.read_text())
    info["total_episodes"] = 251
    path.write_text(json.dumps(info))
    report = run(job)
    assert report["planning_status"] == report["status"] == "PASS"
    assert report["counts"]["declared_episodes_total"] == 251
    assert report["counts"]["episodes_total"] == 2
    assert report["counts"]["selected_windows"] == 6
    assert report["findings"][0]["code"] == "consumer.plan_episode_count_mismatch"
    assert report["findings"][0]["status"] == "WARNING"
