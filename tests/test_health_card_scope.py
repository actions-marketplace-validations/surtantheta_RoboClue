"""Consumer-format requirements require an explicit health-card declaration."""
import copy
import importlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.health_cards import _request, health_card, scan_health_cards, verify_health_card
from policy_doctor.preflight import preflight


@pytest.fixture
def local_v2(tmp_path):
    """Generated Noor-shaped LeRobot input; no claim of native-loader execution."""
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {key: {"dtype": "float32", "shape": [1]} for key in ("action", "observation.state")}
    info = {"codebase_version": "v2.1", "chunks_size": 1000, "fps": 30, "total_frames": 4,
            "total_episodes": 1, "total_tasks": 1,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": features}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/modality.json").write_text(json.dumps({
        kind: {"axis": {"start": 0, "end": 1}} for kind in ("action", "state")}))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 4, "tasks": ["generated"]}) + "\n")
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "generated"}) + "\n")
    rows = [{"index": i, "episode_index": 0, "frame_index": i, "timestamp": i / 30,
             "action": [float(i)], "observation.state": [float(i)]} for i in range(4)]
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/episode_000000.parquet")
    return root


def test_generic_card_omits_unrequested_gr00t_but_preserves_unknown_coverage(local_v2):
    card = health_card(str(local_v2))
    assert card["request"]["config"]["groot_contract"] is False
    assert card["status"] == "unknown"
    assert "groot_contract" not in card["report"]["components"]
    assert not any(f["code"].startswith("groot.") for f in card["report"]["findings"])
    assert card["report"]["components"]["numeric_rows"]["status"] == "pass"
    assert card["report"]["components"]["dataset_statistics"]["status"] == "unknown"
    assert card["report"]["components"]["episode_windows"]["status"] == "unknown"
    assert verify_health_card(card)


@pytest.mark.parametrize("consumer", ["unspecified", "lerobot", "groot", "Isaac-GR00T N1.5"])
def test_consumer_label_never_selects_gr00t_format_requirements(local_v2, consumer):
    card = health_card(str(local_v2), consumer=consumer)
    assert card["request"]["config"]["groot_contract"] is False
    assert card["status"] == "unknown" and "groot_contract" not in card["report"]["components"]


@pytest.mark.parametrize("config", [{"groot_contract": True}, {"groot_version": "n1.5"}])
def test_explicit_contract_or_version_preserves_real_n15_errors(local_v2, config):
    card = health_card(str(local_v2), config=config)
    assert card["request"]["config"]["groot_contract"] is True
    assert card["status"] == "error"
    errors = {f["code"] for f in card["report"]["components"]["groot_contract"]["findings"]
              if f["status"] == "error" and f.get("groot_version") == "n1.5"}
    assert {"groot.metadata", "groot.modality_schema"} <= errors


@pytest.mark.parametrize("selector,value", [("groot_modality_config", "selected.py"),
                                            ("groot_config_name", "selected"),
                                            ("groot_version", "n1.5")])
def test_each_explicit_preflight_dataset_contract_selector_enables_omitted_boolean(selector, value):
    config = {selector: value}
    original = copy.deepcopy(config)
    request = _request("local", "arbitrary label", config, "main", False)
    assert config == original
    assert request["config"]["groot_contract"] is True
    assert request["config"][selector] == value


@pytest.mark.parametrize("selector,value", [("groot_modality_config", None),
                                            ("groot_config_name", None), ("groot_version", None),
                                            ("checkpoint", "selected-checkpoint"),
                                            ("groot_embodiment", "selected-embodiment"),
                                            ("openpi_config", "selected-openpi-config")])
def test_absent_dataset_selectors_and_other_interfaces_do_not_enable_contract(selector, value):
    request = _request("local", "groot", {selector: value}, "main", False)
    assert request["config"]["groot_contract"] is False


@pytest.mark.parametrize("selector", ["groot_modality_config", "groot_config_name", "groot_version"])
def test_explicit_false_with_any_contract_selector_keeps_preflight_refusal(local_v2, tmp_path, selector):
    config_file = tmp_path / "selected.py"
    config_file.write_text("# Configuration data; never executed by this check\n")
    value = {"groot_modality_config": str(config_file), "groot_config_name": "selected", "groot_version": "n1.5"}[selector]
    with pytest.raises(InputError, match="cannot be combined"):
        health_card(str(local_v2), config={"groot_contract": False, selector: value})


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_contract_boolean_remains_strict(local_v2, value):
    with pytest.raises(InputError, match="strict booleans"):
        health_card(str(local_v2), config={"groot_contract": value})


def test_explicit_false_and_default_are_identical_effective_checks(local_v2):
    default = health_card(str(local_v2))
    disabled = health_card(str(local_v2), config={"groot_contract": False})
    assert default["card_id"] == disabled["card_id"]
    assert default["request"]["config"] == disabled["request"]["config"]


def test_preflight_keeps_its_existing_automatic_gr00t_behavior(local_v2):
    report = preflight(str(local_v2))
    assert report["status"] == "error"
    assert report["components"]["groot_contract"]["status"] == "error"


def test_generic_scope_preserves_independent_numeric_error(local_v2):
    path = local_v2 / "data/chunk-000/episode_000000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[1]["action"] = [float("nan")]
    pq.write_table(pa.Table.from_pylist(rows), path)
    card = health_card(str(local_v2))
    assert card["status"] == "error"
    assert "groot_contract" not in card["report"]["components"]
    assert any(f["code"] == "dataset.vector_invalid" for f in card["report"]["findings"])


def test_generic_scope_preserves_independent_video_index_error(local_v2):
    info_path = local_v2 / "meta/info.json"
    info = json.loads(info_path.read_text())
    info.update(codebase_version="v3.0", data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    info["features"]["observation.images.front"] = {"dtype": "video", "shape": [3, 4, 3]}
    info_path.write_text(json.dumps(info))
    (local_v2 / "data/chunk-000/episode_000000.parquet").rename(local_v2 / "data/chunk-000/file-000.parquet")
    episodes = local_v2 / "meta/episodes/chunk-000"
    episodes.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"episode_index": 0, "length": 4, "dataset_from_index": 0,
        "dataset_to_index": 4, "data/chunk_index": 0, "data/file_index": 0}]), episodes / "file-000.parquet")
    card = health_card(str(local_v2))
    assert card["status"] == "error"
    assert "groot_contract" not in card["report"]["components"]
    assert any(f["code"] == "video.feature_without_index" for f in card["report"]["findings"])


def test_stored_explicit_true_card_verification_does_not_reinterpret_defaults(local_v2, monkeypatch):
    card = health_card(str(local_v2), config={"groot_contract": True})
    original = copy.deepcopy(card)
    module = importlib.import_module("policy_doctor.health_cards")
    monkeypatch.setattr(module, "preflight", lambda *a, **k: pytest.fail("Stored-card verification reran preflight"))
    assert verify_health_card(card)
    assert card == original and card["status"] == "error"


def test_batch_retains_effective_scope_per_entry_without_mutating_config(local_v2):
    config = {"groot_version": "n1.5", "max_rows": 4}
    original = copy.deepcopy(config)
    result = scan_health_cards([str(local_v2), {"source": str(local_v2), "config": config}])
    assert config == original
    assert [card["status"] for card in result["cards"]] == ["unknown", "error"]
    assert [card["request"]["config"]["groot_contract"] for card in result["cards"]] == [False, True]
    assert all(verify_health_card(card) for card in result["cards"])
