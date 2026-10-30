"""Generated native-shaped persisted rows, never a native writer execution."""

import hashlib
import json

import pytest
from test_named_actions import change, commit, contract
from test_named_actions import packet as packet_fixture

from policy_doctor import named_actions
from policy_doctor.artifacts import InputError

DATA_FILES = (
    "data/chunk-000/file-000.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
)


@pytest.fixture
def recording(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    base = packet_fixture.__wrapped__(tmp_path, monkeypatch)

    def create(name="recording", *, limited=True, recorded_corrected=False, offset=0.0):
        root = base(name, limited=limited, offset=offset)
        data = json.loads((root / "capture.json").read_text())
        data["schema_version"] = 2
        sources = {
            key: ("generated recording native module " + key).encode()
            for key in ("episodic.py", "feature_utils.py", "lerobot_dataset.py", "dataset_writer.py")
        }
        monkeypatch.setattr(
            named_actions,
            "RECORDING_SOURCE_HASHES",
            {key: hashlib.sha256(raw).hexdigest() for key, raw in sources.items()},
        )
        for key, raw in sources.items():
            (root / "sources" / key).write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            data["source"]["files"][key] = {"sha256": digest}
            data["files"].append({"path": "sources/" + key, "sha256": digest})
        tick = data["ticks"][0]
        recorded = dict(tick["driver_return"] if recorded_corrected else tick["canonical_at_return"])
        tick["canonical_at_return"] = dict(recorded)
        actions = [recorded[name] for name in data["action_names"]]
        rows = [
            {
                "tick_id": 0,
                "episode_index": 0,
                "frame_index": 0,
                "index": 0,
                "source_role": "canonical_return",
                "action_names": list(data["action_names"]),
                "canonical_at_frame": dict(recorded),
                "frame_action": list(actions),
                "action": list(actions),
                "buffer_action": list(actions),
                "facade_completed": True,
                "writer_completed": True,
                "buffer_size_before": 0,
                "buffer_size_after": 1,
                "relations": {
                    key: True
                    for key in (
                        "canonical_to_build_frame",
                        "build_to_facade",
                        "facade_to_writer",
                        "writer_to_buffer",
                    )
                },
            }
        ]
        lifecycle = {
            key: True
            for key in (
                "loop_completed",
                "facade_save_completed",
                "writer_save_completed",
                "facade_finalize_completed",
                "writer_finalize_completed",
            )
        }
        lifecycle.update(save_episode_count=1, finalize_count=1, save_buffer_continuity=True)
        data["recording"] = {
            "coverage": {"target": "same_native_episodic_writer", "complete_target_call_history": True},
            "lifecycle": lifecycle,
            "rows": rows,
            "dataset_files": [],
        }
        dataset = root / "dataset"
        for path in DATA_FILES:
            (dataset / path).parent.mkdir(parents=True, exist_ok=True)
        info = {
            "codebase_version": "v3.0",
            "storage_format": "parquet",
            "total_frames": 1,
            "total_episodes": 1,
            "total_tasks": 1,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": {"action": {"dtype": "float32", "shape": [2], "names": list(data["action_names"])}},
        }
        (dataset / "meta/info.json").write_text(json.dumps(info))
        (dataset / "meta/stats.json").write_text("{}")
        pq.write_table(
            pa.table(
                {
                    "index": [0],
                    "episode_index": [0],
                    "frame_index": [0],
                    "action": pa.array([actions], type=pa.list_(pa.float32(), 2)),
                }
            ),
            dataset / DATA_FILES[0],
        )
        pq.write_table(
            pa.table(
                {
                    "episode_index": [0],
                    "dataset_from_index": [0],
                    "dataset_to_index": [1],
                    "length": [1],
                    "data/chunk_index": [0],
                    "data/file_index": [0],
                }
            ),
            dataset / DATA_FILES[1],
        )
        pq.write_table(pa.table({"task_index": [0]}), dataset / DATA_FILES[4])
        for path in DATA_FILES:
            entry = {
                "path": "dataset/" + path,
                "sha256": hashlib.sha256((dataset / path).read_bytes()).hexdigest(),
            }
            data["recording"]["dataset_files"].append(entry)
            data["files"].append(dict(entry))
        commit(root, data)
        return root

    return create


def test_recorded_canonical_divergence_is_observation_then_explicit_fault(recording):
    root = recording()
    report = named_actions.audit_named_actions(root, **contract())
    assert report["status"] == "pass"
    assert report["recorded_action_status"] == "pass"
    assert report["recording"]["driver_difference_events"] == 1
    strict = named_actions.audit_named_actions(
        root, **contract(), require_recorded_matches_driver_return=True
    )
    assert strict["status"] == strict["acceptance_contract"]["status"] == "error"
    violation = strict["acceptance_contract"]["violations"][0]
    assert violation["field"] == "recorded_matches_driver_return"
    assert any(e.get("column") == "action" and e["row"] == 0 for e in violation["evidence"])


def test_actual_supplied_recording_correction_allows_new_values(recording):
    for offset in (0.0, 20.0):
        root = recording("healthy" + str(offset), recorded_corrected=True, offset=offset)
        report = named_actions.audit_named_actions(
            root, **contract(), require_recorded_matches_driver_return=True
        )
        assert report["status"] == report["recording"]["status"] == "pass"
        assert report["recording"]["rows_checked"] == 1


@pytest.mark.parametrize("mode", ["coverage", "save", "finalize", "relation", "buffer", "tick", "canonical"])
def test_incomplete_or_wrong_join_cannot_claim_recorded_equality(recording, mode):
    root = recording(recorded_corrected=True)

    def mutate(d):
        rec = d["recording"]
        if mode == "coverage":
            rec["coverage"]["complete_target_call_history"] = False
        elif mode == "save":
            rec["lifecycle"]["writer_save_completed"] = False
        elif mode == "finalize":
            rec["lifecycle"]["writer_finalize_completed"] = False
        elif mode == "relation":
            rec["rows"][0]["relations"]["facade_to_writer"] = False
        elif mode == "buffer":
            rec["rows"][0]["buffer_action"][0] += 1.0
        elif mode == "tick":
            rec["rows"][0]["tick_id"] = 10
        else:
            rec["rows"][0]["canonical_at_frame"]["a.pos"] += 1.0

    change(root, mutate)
    result = named_actions.audit_named_actions(
        root, **contract(), require_recorded_matches_driver_return=True
    )
    assert result["acceptance_contract"]["status"] == "unknown"


def test_legacy_capture_recording_criterion_remains_unknown(tmp_path, monkeypatch):
    root = packet_fixture.__wrapped__(tmp_path, monkeypatch)()
    result = named_actions.audit_named_actions(
        root, **contract(), require_recorded_matches_driver_return=True
    )
    assert result["status"] == result["recorded_action_status"] == "unknown"


@pytest.mark.parametrize("value", [True, 0, 1, "yes", None])
def test_recording_criterion_strict_bool(recording, value):
    root = recording()
    if value is True:
        assert (
            named_actions.audit_named_actions(root, require_recorded_matches_driver_return=value)["status"]
            == "error"
        )
    else:
        with pytest.raises(InputError, match="Boolean"):
            named_actions.audit_named_actions(root, require_recorded_matches_driver_return=value)


def refresh_dataset_file(root, relative, mutate):
    path = root / relative
    mutate(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def update(data):
        for entry in data["files"] + data["recording"]["dataset_files"]:
            if entry["path"] == relative:
                entry["sha256"] = digest

    change(root, update)


def test_optional_arrow_absence_only_disables_persistence(recording, monkeypatch):
    import builtins

    root = recording(recorded_corrected=True)
    original = builtins.__import__

    def no_arrow(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("controlled optional dependency absence")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_arrow)
    report = named_actions.audit_named_actions(root, **contract())
    assert report["status"] == "pass" and report["recording"]["persistence_status"] == "unknown"
    report = named_actions.audit_named_actions(
        root, **contract(), require_recorded_matches_driver_return=True
    )
    assert report["status"] == "unknown" and "Install" in report["recording"]["reason"]


@pytest.mark.parametrize(
    "mode", ["missing_continuity", "source_role", "duplicate_rows", "metadata", "row_identity", "row_values"]
)
def test_persistence_or_origin_mismatch_remains_unknown(recording, mode):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = recording(recorded_corrected=True)
    if mode == "missing_continuity":
        change(root, lambda d: d["recording"]["lifecycle"].pop("save_buffer_continuity"))
    elif mode == "source_role":
        change(root, lambda d: d["recording"]["rows"][0].update(source_role="driver_return"))
    elif mode == "duplicate_rows":
        change(root, lambda d: d["recording"]["rows"].append(dict(d["recording"]["rows"][0])))
    elif mode == "metadata":

        def modify(path):
            value = json.loads(path.read_text())
            value["total_frames"] = 2
            path.write_text(json.dumps(value))

        refresh_dataset_file(root, "dataset/meta/info.json", modify)
    else:

        def modify(path):
            data = pq.read_table(path).to_pydict()
            if mode == "row_identity":
                data["index"] = [1]
            else:
                data["action"][0][0] += 1.0
            pq.write_table(pa.table(data), path)

        refresh_dataset_file(root, "dataset/data/chunk-000/file-000.parquet", modify)
    result = named_actions.audit_named_actions(
        root, **contract(), require_recorded_matches_driver_return=True
    )
    assert result["status"] == "unknown" and result["recording"]["status"] == "unknown"


def test_incomplete_lifecycle_without_dataset_copy_retains_buffer_only(recording):
    import shutil

    root = recording(recorded_corrected=True)
    shutil.rmtree(root / "dataset")

    def mutate(data):
        data["files"] = [entry for entry in data["files"] if not entry["path"].startswith("dataset/")]
        data["recording"]["dataset_files"] = []
        data["recording"]["lifecycle"]["writer_finalize_completed"] = False

    change(root, mutate)
    result = named_actions.audit_named_actions(
        root, **contract(), require_recorded_matches_driver_return=True
    )
    assert result["status"] == "unknown"
    assert result["recording"]["buffer_binding_status"] == "pass"
    assert result["recording"]["persistence_status"] == "unknown"


def test_parquet_footer_bound_checked_before_arrow_allocation(recording, monkeypatch):
    import pyarrow.parquet as pq

    root = recording()

    def modify(path):
        raw = path.read_bytes()
        path.write_bytes(raw[:-8] + (2 * 1024 * 1024).to_bytes(4, "little") + b"PAR1")

    refresh_dataset_file(root, "dataset/data/chunk-000/file-000.parquet", modify)

    def forbidden(*args, **kwargs):
        raise AssertionError("ParquetFile allocated before footer preflight")

    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    result = named_actions.audit_named_actions(root, require_recorded_matches_driver_return=True)
    assert result["status"] == "unknown" and "footer" in result["recording"]["reason"]


def test_dataset_tamper_or_escape_refused_before_decode(recording, tmp_path):
    root = recording()
    target = root / "dataset/meta/stats.json"
    target.write_text('{"tampered":1}')
    with pytest.raises(InputError, match="hash"):
        named_actions.audit_named_actions(root)
    root = recording("symlink")
    target = root / "dataset/meta/stats.json"
    target.unlink()
    target.symlink_to(tmp_path / "outside")
    with pytest.raises(InputError, match="symlink"):
        named_actions.audit_named_actions(root)


def test_same_existing_cli_lifecycle_preserves_recording_criterion_and_new_values(
    recording, tmp_path, capsys
):
    import shutil

    from policy_doctor.cli import main

    before, after = recording("before"), recording("after", recorded_corrected=True)
    healthy, fault, unknown = (
        recording("later", recorded_corrected=True, offset=10.0),
        recording("fault", offset=10.0),
        recording("unknown"),
    )
    change(unknown, lambda d: d["recording"]["lifecycle"].update(save_buffer_continuity=False))
    criterion = tmp_path / "criterion.json"
    criterion.write_text(json.dumps({"require_recorded_matches_driver_return": True}))
    case = tmp_path / "case"
    assert (
        main(
            [
                "create-case",
                "native-named-action",
                str(before),
                str(case),
                "--corrected",
                str(after),
                "--named-contract",
                str(criterion),
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["checks"]["before"]["acceptance_contract"]["status"] == "error"
    assert result["checks"]["before"]["recording"]["persistence_status"] == "pass"
    manifest = json.loads((case / "manifest.json").read_text())
    assert manifest["parameters"]["require_recorded_matches_driver_return"] is True
    assert manifest["parameters"]["require_canonical_matches_driver_return"] is False
    moved = tmp_path / "colleague"
    case.rename(moved)
    shutil.rmtree(before)
    shutil.rmtree(after)
    assert main(["reproduce-case", str(moved), "--json"]) == 0
    capsys.readouterr()
    for candidate, code in [(healthy, 0), (fault, 1), (unknown, 3)]:
        assert main(["check-case", str(moved), str(candidate), "--json"]) == code
        capsys.readouterr()


def test_same_mcp_contract_and_creation_with_strict_recording_bool(recording, tmp_path):
    pytest.importorskip("mcp")
    from policy_doctor.server import create_server

    before, after = recording("before"), recording("after", recorded_corrected=True)
    output = tmp_path / "output"
    output.mkdir()
    tools = create_server([str(tmp_path)], case_root=str(output))._tool_manager._tools
    assert len(tools) == 31
    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
    criterion = {"require_recorded_matches_driver_return": True}
    assert tools["audit_runtime_capture"].fn(str(before), named_contract=criterion)["status"] == "error"
    with pytest.raises(InputError, match="Boolean"):
        tools["audit_runtime_capture"].fn(
            str(before), named_contract={"require_recorded_matches_driver_return": 1}
        )
    case = output / "case"
    assert (
        tools["create_incident_case"].fn(
            "native_named_action", str(before), str(case), criterion, corrected=str(after)
        )["status"]
        == "pass"
    )
    assert tools["check_incident_candidate"].fn(str(case), str(after))["status"] == "pass"


def test_native_scalar_action_shape_one_is_supported(recording):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = recording(recorded_corrected=True)
    def scalar(data):
        data['action_names'] = data['engine_action_names'] = ['a.pos']
        tick = data['ticks'][0]
        tick['interpolated'] = tick['interpolated'][:1]
        for key in named_actions.STAGES:
            tick[key] = {'a.pos': tick[key]['a.pos']}
        row = data['recording']['rows'][0]
        row['action_names'] = ['a.pos']
        row['canonical_at_frame'] = {'a.pos': row['canonical_at_frame']['a.pos']}
        for key in ('frame_action', 'action', 'buffer_action'):
            row[key] = row[key][:1]
    change(root, scalar)
    def info(path):
        data = json.loads(path.read_text())
        data['features']['action'].update(shape=[1], names=['a.pos'])
        path.write_text(json.dumps(data))
    refresh_dataset_file(root, 'dataset/meta/info.json', info)
    def rows(path):
        table = pa.table({'index': [0], 'episode_index': [0], 'frame_index': [0], 'action': pa.array([1.], type=pa.float32())})
        pq.write_table(table, path)
    refresh_dataset_file(root, 'dataset/data/chunk-000/file-000.parquet', rows)
    report = named_actions.audit_named_actions(root, require_recorded_matches_driver_return=True)
    assert report['status'] == 'pass' and report['recording']['rows_checked'] == 1


def test_native_feature_order_drives_saved_named_mapping(recording):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = recording(recorded_corrected=True)
    def reorder(data):
        row = data['recording']['rows'][0]
        row['action_names'].reverse()
        for key in ('frame_action', 'action', 'buffer_action'):
            row[key].reverse()
    change(root, reorder)
    def info(path):
        data = json.loads(path.read_text())
        data['features']['action']['names'].reverse()
        path.write_text(json.dumps(data))
    refresh_dataset_file(root, 'dataset/meta/info.json', info)
    def rows(path):
        data = pq.read_table(path).to_pydict()
        data['action'][0].reverse()
        data['action'] = pa.array(data['action'], type=pa.list_(pa.float32(), 2))
        pq.write_table(pa.table(data), path)
    refresh_dataset_file(root, 'dataset/data/chunk-000/file-000.parquet', rows)
    assert named_actions.audit_named_actions(root, **contract(), require_recorded_matches_driver_return=True)['status'] == 'pass'
