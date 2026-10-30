"""Negative-control intake: non-robot sensor captures (SiGNgapore-like iPhone recordings) and non-dataset folders."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from policy_doctor import preflight as engine
from policy_doctor.artifacts import InputError
from policy_doctor.cli import human
from policy_doctor.investigation import NO_ARTIFACT_INCOMPLETE, NO_SUPPORTED_ARTIFACT, investigate_run
from policy_doctor.reports import markdown_report
from policy_doctor.server import SUMMARY_LIMIT, _summary

SRC = Path(__file__).resolve().parents[1] / "src"
ENV = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")


def cli(*args):
    return subprocess.run([sys.executable, "-B", "-m", "policy_doctor", *map(str, args)], capture_output=True,
                          text=True, env=ENV, timeout=300)


def phone_capture(root):
    """The layout of a StrayScanner capture: camera matrix, IMU and odometry CSVs, video, depth archives, VQA JSON."""
    (root / "VQA").mkdir(parents=True)
    (root / "camera_matrix.csv").write_text("1514.2687, 0.0, 950.4587\n0.0, 1514.2687, 729.12415\n0.0, 0.0, 1.0\n")
    rows = ["timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z"]
    rows += [f"{101170.6 + i / 100}, -8.99, -0.30, -2.24, 0.02, 0.04, 0.06" for i in range(50)]
    (root / "imu.csv").write_text("\n".join(rows) + "\n")
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw"]
    rows += [f"{101170.5 + i / 15}, {i:06d}, 0.1, 0.2, 0.03, -0.14, 0.14, -0.97, -0.02" for i in range(20)]
    (root / "odometry.csv").write_text("\n".join(rows) + "\n")
    (root / "rgb.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    (root / "depth.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    (root / "VQA" / "parsing_gt.json").write_text(json.dumps([{"sign": "exit"}]))
    (root / "VQA" / "query.json").write_text(json.dumps({"question": "where is the exit?"}))
    return root


def test_sensor_capture_has_no_recordings_and_says_so(tmp_path):
    root = phone_capture(tmp_path / "S9")
    result = investigate_run(str(root))
    assert result["inventory"]["recordings"] == [] and result["no_supported_artifact"] is True
    assert result["summary"].startswith(NO_SUPPORTED_ARTIFACT) and "0 recording(s)" in result["summary"]
    assert "3 tabular file(s) are not rollout recordings" in result["summary"]
    unrecognized = result["coverage"]["tabular_files_unrecognized"]
    assert [Path(item["path"]).name for item in unrecognized] == ["camera_matrix.csv", "imu.csv", "odometry.csv"]
    assert all("without an action stream" in item["reason"] for item in unrecognized)
    assert "streams: timestamp, a_x, a_y" in unrecognized[1]["reason"]
    assert result["coverage"]["tabular_files_unrecognized_count"] == 3
    assert result["coverage"]["tabular_files_not_inspected_count"] == 0
    assert not any(a["tool"] == "compare_rollout_recordings" for a in result["next_actions"])
    assert result["status"] == "unknown"
    text = human(result)
    assert "No supported robot artifact was found here" in text
    assert 'Not a recognized rollout recording: "' + str(root / "camera_matrix.csv") + '"' in text
    assert "compare_rollout_recordings" not in text
    markdown = markdown_report(result)
    assert "## Tabular files not recognized as rollout recordings" in markdown and "- recordings: 0" in markdown
    concise = _summary(result)
    assert concise["discovery"]["no_supported_artifact"] is True and concise["discovery"]["recordings"] == 0
    assert concise["discovery"]["tabular_files_unrecognized"] == 3
    assert len(json.dumps(concise).encode()) <= SUMMARY_LIMIT


def test_recognized_rollouts_are_counted_and_compared_only_in_pairs(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    np.savez(root / "first.npz", step=np.arange(4), action=np.zeros((4, 2)))
    (root / "imu.csv").write_text("timestamp,a_x\n0,1\n1,2\n")
    single = investigate_run(str(root))
    assert [Path(r["path"]).name for r in single["inventory"]["recordings"]] == ["first.npz"]
    assert not any(a["tool"] == "compare_rollout_recordings" for a in single["next_actions"])
    assert single["no_supported_artifact"] is False and not single["summary"].startswith(NO_SUPPORTED_ARTIFACT)
    assert "1 tabular file(s) are not rollout recordings" in single["summary"]
    (root / "second.csv").write_text("step,action_0,action_1\n0,0.0,0.0\n1,0.5,0.1\n")
    pair = investigate_run(str(root))
    assert [Path(r["path"]).name for r in pair["inventory"]["recordings"]] == ["first.npz", "second.csv"]
    compare = next(a for a in pair["next_actions"] if a["tool"] == "compare_rollout_recordings")
    assert [Path(p).name for p in compare["candidates"]] == ["first.npz", "second.csv"]
    assert [Path(i["path"]).name for i in pair["coverage"]["tabular_files_unrecognized"]] == ["imu.csv"]


@pytest.mark.parametrize("names,recognized", [
    (["action"], True), (["actions"], True), (["action.joint_0"], True), (["joint_action"], True),
    (["cmd actions"], True), (["transaction"], False), (["factions"], False), (["timestamp", "x"], False)])
def test_action_stream_names(tmp_path, names, recognized):
    root = tmp_path / "run"
    root.mkdir()
    np.savez(root / "r.npz", **{name: np.zeros(3) for name in names})
    result = investigate_run(str(root))
    assert bool(result["inventory"]["recordings"]) is recognized
    assert bool(result["coverage"]["tabular_files_unrecognized"]) is not recognized


def test_large_tabular_files_are_judged_by_stream_names_without_decoding(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    np.savez(root / "big_rollout.npz", step=np.arange(300_000), action=np.zeros(300_000))  # over the 1 MiB bound
    np.savez(root / "big_sensor.npz", timestamp=np.zeros(300_000), a_x=np.zeros(300_000))
    (root / "big_imu.csv").write_text("timestamp, a_x, a_y\n" + "0.0, 1.0, 2.0\n" * 120_000)
    (root / "big_commands.csv").write_text("step,action_0\n" + "0,0.5\n" * 300_000)
    (root / "big_broken.npz").write_bytes(b"\x01" * (1024 * 1024 + 10))
    (root / "big_native.parquet").write_bytes(b"\x00" * (1024 * 1024 + 10))
    (root / "broken.npz").write_bytes(b"not a zip")
    result = investigate_run(str(root))
    recordings = {Path(r["path"]).name: r for r in result["inventory"]["recordings"]}
    assert sorted(recordings) == ["big_commands.csv", "big_native.parquet", "big_rollout.npz"]
    assert all(r["inventory_not_read"] and "inventory" not in r for r in recordings.values())
    unrecognized = {Path(i["path"]).name: i["reason"] for i in result["coverage"]["tabular_files_unrecognized"]}
    assert sorted(unrecognized) == ["big_imu.csv", "big_sensor.npz", "broken.npz"]
    assert "streams from the archive directory, not decoded: timestamp, a_x" in unrecognized["big_sensor.npz"]
    assert "streams from the header line, not decoded: timestamp, a_x, a_y" in unrecognized["big_imu.csv"]
    assert "NPZ" in unrecognized["broken.npz"]
    big = result["coverage"]["tabular_files_not_inspected"]
    assert [Path(i["path"]).name for i in big] == ["big_broken.npz"] and "1 MiB" in big[0]["reason"]
    assert "compare_rollout_recordings directly" in big[0]["reason"]
    assert "1 tabular file(s) were not inspected" in result["summary"]
    compare = next(a for a in result["next_actions"] if a["tool"] == "compare_rollout_recordings")
    assert [Path(p).name for p in compare["candidates"]] == ["big_commands.csv", "big_native.parquet", "big_rollout.npz"]
    text = human(result)
    assert 'Not inspected: "' + str(root / "big_broken.npz") + '"' in text


def test_no_artifact_statement_is_withheld_when_discovery_was_incomplete(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "big_broken.npz").write_bytes(b"\x01" * (1024 * 1024 + 10))
    result = investigate_run(str(root))
    assert result["no_supported_artifact"] is False
    assert result["summary"].startswith(NO_ARTIFACT_INCOMPLETE) and NO_SUPPORTED_ARTIFACT not in result["summary"]
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "elsewhere.npz").symlink_to(root / "big_broken.npz")
    result = investigate_run(str(linked))
    assert result["coverage"]["symlinks_skipped"] == 1 and result["no_supported_artifact"] is False
    assert result["summary"].startswith(NO_ARTIFACT_INCOMPLETE)
    empty = tmp_path / "empty"
    empty.mkdir()
    result = investigate_run(str(empty))
    assert result["no_supported_artifact"] is True and result["summary"].startswith(NO_SUPPORTED_ARTIFACT)
    assert _summary(result)["discovery"]["no_supported_artifact"] is True


@pytest.mark.parametrize('layout', ['wide_csv', 'long_csv', 'wide_npz'])
def test_capped_header_names_never_establish_complete_no_artifact_absence(tmp_path, layout):
    root = tmp_path / 'run'
    root.mkdir()
    if layout == 'wide_csv':
        names = [f'sensor_{i}' for i in range(4096)] + ['action_0']
        (root / 'wide.csv').write_text(','.join(names) + '\n' + (','.join(['0'] * len(names)) + '\n') * 150)
    elif layout == 'long_csv':
        (root / 'long.csv').write_text('sensor_' + 'x' * (64 * 1024) + ',action_0\n' + '0,0\n' * 300000)
    else:
        names = {f'sensor_{i}': np.zeros(64, dtype=np.float32) for i in range(4096)}
        names['action'] = np.zeros(64, dtype=np.float32)
        np.savez(root / 'wide.npz', **names)
    result = investigate_run(str(root))
    assert result['no_supported_artifact'] is False and result['summary'].startswith(NO_ARTIFACT_INCOMPLETE)
    assert result['inventory']['recordings'] == []
    assert result['coverage']['tabular_files_not_inspected_count'] == 1
    assert result['coverage']['tabular_files_unrecognized_count'] == 0
    assert 'not be completely read' in result['coverage']['tabular_files_not_inspected'][0]['reason']


def test_complete_header_at_the_name_bound_still_recognizes_an_action_candidate(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    names = [f'sensor_{i}' for i in range(4095)] + ['action_0']
    (root / 'wide.csv').write_text(','.join(names) + '\n' + (','.join(['0'] * len(names)) + '\n') * 150)
    result = investigate_run(str(root))
    assert len(result['inventory']['recordings']) == 1
    assert result['coverage']['tabular_files_not_inspected_count'] == 0
    assert result['no_supported_artifact'] is False


def test_native_parquet_needing_an_episode_selection_is_a_recording(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "run"
    root.mkdir()
    for name in ("a", "b"):
        pq.write_table(pa.table({"episode_index": [0, 0, 1, 1], "action": [[0.0, 1.0]] * 4}), root / f"{name}.parquet")
    result = investigate_run(str(root))
    assert len(result["inventory"]["recordings"]) == 2
    assert all("Multiple native episodes" in r["needs_selection_or_adapter"] for r in result["inventory"]["recordings"])
    assert any(a["tool"] == "compare_rollout_recordings" for a in result["next_actions"])


def test_files_inside_a_dataset_are_not_listed_as_other_tabular_files(tmp_path):
    import test_preflight as fixtures

    root = tmp_path / "run"
    ds = fixtures.dataset(root)
    (ds / "meta" / "extra.csv").write_text("timestamp,x\n0,1\n")
    result = investigate_run(str(root))
    assert result["coverage"]["tabular_files_unrecognized_count"] == 0
    assert result["no_supported_artifact"] is False


# ---------------------------------------------------------------- preflight on a folder that is not a dataset

def test_preflight_leads_with_not_a_lerobot_dataset(tmp_path):
    root = phone_capture(tmp_path / "S9")
    report = engine.preflight(str(root))
    assert report["summary"] == engine.NOT_A_DATASET and report["status"] == "unknown"
    assert report["next_actions"][0] == engine.NOT_A_DATASET_ACTION
    assert not any("raise the stated budget" in a or "install the requested extra" in a for a in report["next_actions"])
    assert report["coverage"]["not_a_lerobot_dataset"] is True
    assert human(report).splitlines()[1] == engine.NOT_A_DATASET
    assert _summary(report)["headline"] == engine.NOT_A_DATASET
    out = cli("preflight", root)
    assert out.returncode == 3 and out.stdout.splitlines()[1] == engine.NOT_A_DATASET
    assert "raise the stated budget" not in out.stdout
    (root / "config.json").write_text(json.dumps({"type": "act"}))
    assert "may be a policy checkpoint" in engine.preflight(str(root))["summary"]
    missing = tmp_path / "absent"
    with pytest.raises(InputError, match="Dataset source does not exist") as refused:
        engine.preflight(str(missing))
    assert str(missing) in str(refused.value)


def test_a_real_dataset_keeps_its_ordinary_summary(tmp_path):
    import test_preflight as fixtures

    report = engine.preflight(str(fixtures.dataset(tmp_path / "ds")))
    assert report["summary"] != engine.NOT_A_DATASET and "not_a_lerobot_dataset" not in report["coverage"]
    assert engine.NOT_A_DATASET_ACTION not in report["next_actions"]


def test_missing_investigation_directory_is_named_in_library_and_cli_refusals(tmp_path):
    missing = str(tmp_path / 'missing run directory')
    with pytest.raises(InputError, match='requires an existing run directory') as refused:
        investigate_run(missing)
    assert repr(missing) in str(refused.value)
    out = cli('investigate', missing)
    assert out.returncode == 2 and missing in out.stderr


def test_actual_stdio_server_for_a_sensor_capture(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = phone_capture(tmp_path / "S9")

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-B", "-m", "policy_doctor", "serve", "--root",
                                                                     str(tmp_path)], env=ENV)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                detailed = (await client.call_tool("investigate_run_directory", {"directory": str(root)})).structuredContent
                assert detailed["no_supported_artifact"] is True and detailed["inventory"]["recordings"] == []
                assert detailed["coverage"]["tabular_files_unrecognized_count"] == 3
                concise = (await client.call_tool("investigate_run_directory", {
                    "directory": str(root), "response_format": "concise"})).structuredContent
                assert concise["discovery"]["summary"].startswith(NO_SUPPORTED_ARTIFACT)
                assert not any("compare_rollout_recordings" in a for a in concise.get("next_actions", []))
                pre = (await client.call_tool("preflight_check", {"source": str(root)})).structuredContent
                assert pre["headline"] == engine.NOT_A_DATASET
                assert pre["next_actions"][0] == engine.NOT_A_DATASET_ACTION
    asyncio.run(asyncio.wait_for(run(), timeout=120))
