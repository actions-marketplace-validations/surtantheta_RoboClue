"""Health cards preserve actual dataset findings and never promote missing evidence."""

import copy
import hashlib
import importlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError

cards = importlib.import_module("policy_doctor.health_cards")


def dataset(tmp_path, *, broken=False):
    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v3.0", "fps": 30, "total_frames": 6, "total_episodes": 1,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": {"action": {"dtype": "float32", "shape": [1], "names": ["a"]},
                         "observation.state": {"dtype": "float32", "shape": [1], "names": ["a"]}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = [{"index": i, "episode_index": 0, "frame_index": i, "timestamp": i / 30,
             "action": [float(i)], "observation.state": [float(i)]}
            for i in range(6)]
    statistics = {}
    for key in info["features"]:
        values = np.asarray([row[key] for row in rows], dtype=np.float32)
        statistics[key] = {name: value.tolist() for name, value in {
            "mean": values.mean(0), "std": values.std(0), "min": values.min(0),
            "max": values.max(0), "count": np.asarray([len(rows)])}.items()}
    (root / "meta/stats.json").write_text(json.dumps(statistics))
    if broken:
        rows[2]["action"][0] = float("nan")
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([{"episode_index": 0, "length": 6, "dataset_from_index": 0,
        "dataset_to_index": 6, "data/chunk_index": 0, "data/file_index": 0}]),
        root / "meta/episodes/chunk-000/file-000.parquet")
    return root


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_real_local_pass_binds_effective_config_scope_consumer_and_artifacts(tmp_path):
    root = dataset(tmp_path)
    before = hashes(root)
    card = cards.health_card(str(root), consumer="my-job", config={"max_rows": 6})
    assert card["status"] == "pass" and cards.verify_health_card(card)
    assert card["request"]["config"]["video_headers"] is True
    assert card["request"]["config"]["max_rows"] == 6
    assert card["request"]["consumer"] == "my-job"
    assert card["source_revision"]["kind"] == "observed_artifacts_sha256"
    assert card["report"]["components"]["numeric_rows"]["coverage"]["rows_scanned"] == 6
    assert card["report"]["deployment_status"] == "not_assessed"
    assert "not a whole-directory" in card["source_revision"]["scope"]
    assert hashes(root) == before
    again = cards.health_card(str(root), consumer="my-job", config={"max_rows": 6})
    assert card["card_id"] == again["card_id"]
    for change in ({"consumer": "other-job", "config": {"max_rows": 6}},
                   {"consumer": "my-job", "config": {"max_rows": 7}},
                   {"consumer": "my-job", "config": {"max_rows": 6, "groot_contract": True}}):
        assert cards.health_card(str(root), **change)["card_id"] != card["card_id"]


def test_real_known_defect_remains_error_with_partial_other_checks(tmp_path):
    card = cards.health_card(str(dataset(tmp_path, broken=True)), config={"max_rows": 3})
    assert card["status"] == "error"
    assert card["report"]["components"]["episode_windows"]["status"] == "unknown"
    assert any(f["code"] == "dataset.vector_invalid" for f in card["report"]["findings"])
    assert 'fill="#b3261e"' in cards.render_badge(card)


def test_partial_rows_missing_stats_and_unsupported_input_never_green(tmp_path):
    root = dataset(tmp_path)
    partial = cards.health_card(str(root), config={"max_rows": 2})
    assert partial["status"] == "unknown"
    (root / "meta/stats.json").unlink()
    missing = cards.health_card(str(root))
    assert missing["status"] == "unknown"
    unknown_root = tmp_path / "unrecognized"
    unknown_root.mkdir()
    unsupported = cards.health_card(str(unknown_root))
    assert unsupported["status"] == "unknown"
    for card in (partial, missing, unsupported):
        assert 'fill="#276b37"' not in cards.render_badge(card)
        assert cards.verify_health_card(card)


@pytest.mark.parametrize("field", ["scope", "request", "source_revision", "report", "status"])
def test_tampered_card_is_refused_before_output(tmp_path, field):
    card = cards.health_card(str(dataset(tmp_path)))
    card[field] = "changed"
    assert not cards.verify_health_card(card)
    with pytest.raises(InputError, match="binding"):
        cards.write_health_card(card, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_changed_content_changes_local_revision(tmp_path):
    root = dataset(tmp_path)
    before = cards.health_card(str(root))
    stats = root / "meta/stats.json"
    stats.write_text(stats.read_text() + "\n")
    after = cards.health_card(str(root))
    assert before["status"] == after["status"] == "pass"
    assert before["source_revision"] != after["source_revision"]


@pytest.mark.parametrize("mutation", ["missing_default", "missing_values"])
def test_omitted_effective_config_is_refused_even_if_locally_resealed(tmp_path, mutation):
    card = cards.health_card(str(dataset(tmp_path)))
    if mutation == "missing_default":
        del card["request"]["config"]["video_headers"]
    else:
        card["request"]["config"] = sorted(cards.CONFIG_KEYS)
    payload = {key: value for key, value in card.items() if key != "card_id"}
    card["card_id"] = "sha256:" + cards._digest(payload)
    assert not cards.verify_health_card(card)
    with pytest.raises(InputError, match="binding"):
        cards.render_badge(card)


def test_badge_xml_is_escaped_and_retains_bounded_claim(tmp_path):
    card = cards.health_card(str(dataset(tmp_path)), consumer='job <A> & "B"')
    svg = cards.render_badge(card)
    root = ET.fromstring(svg)
    assert root.attrib["aria-label"] == "bounded checks: pass"
    assert "job <A> &" in root.find("{http://www.w3.org/2000/svg}title").text
    assert "No whole-dataset quality" in svg
    assert card["card_id"] in svg


def test_ci_outputs_are_valid_and_existing_directory_is_preserved(tmp_path):
    card = cards.health_card(str(dataset(tmp_path)))
    output = tmp_path / "ci"
    paths = cards.write_health_card(card, output)
    assert cards.verify_health_card(json.loads((output / "health-card.json").read_text()))
    assert set(paths) == {"report", "badge", "summary"}
    before = hashes(output)
    with pytest.raises(FileExistsError):
        cards.write_health_card(card, output)
    assert hashes(output) == before


@pytest.mark.parametrize("options", [{"fetch": 1}, {"consumer": ""}, {"consumer": "a" * 129},
                                     {"config": {"fetch": True}}, {"config": {"max_rows": float("nan")}},
                                     {"revision": "a" * 40}])
def test_invalid_local_request_is_rejected_before_checker(monkeypatch, options):
    monkeypatch.setattr(cards, "preflight", lambda *a, **k: pytest.fail("checker reached"))
    with pytest.raises(InputError):
        cards.health_card("local", **options)


@pytest.mark.parametrize("source", ["hf://org/data", "https://example.test/dataset", "hf://org/../data"])
def test_network_and_invalid_hf_sources_are_refused_before_checker(monkeypatch, source):
    monkeypatch.setattr(cards, "preflight", lambda *a, **k: pytest.fail("checker reached"))
    with pytest.raises(InputError):
        cards.health_card(source)


def test_explicit_public_pinned_intake_retains_revision_and_error(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    engine = importlib.import_module("policy_doctor.preflight")
    root = dataset(tmp_path, broken=True)
    sha = "a" * 40
    calls = []

    @contextmanager
    def intake(source, **kwargs):
        calls.append((source, kwargs))
        yield SimpleNamespace(root=root, provenance={"source": source, "resolved_revision": sha,
            "files": [{"path": name, "sha256": digest} for name, digest in hashes(root).items()]},
            video_context=None)

    monkeypatch.setattr(engine.hub, "resolve_dataset_content", intake)
    card = cards.health_card("hf://org/data", revision=sha, fetch=True)
    assert card["status"] == "error"
    assert card["source_revision"] == {"kind": "hub_commit", "value": sha}
    assert len(calls) == 1 and calls[0][1]["revision"] == sha
    assert str(root) not in json.dumps(card)
    assert cards.verify_health_card(card)


def test_timeout_is_unknown_not_intrinsic_dataset_defect(tmp_path, monkeypatch):
    engine = importlib.import_module("policy_doctor.preflight")

    def timeout(*args, **kwargs):
        raise TimeoutError("selected public intake timed out")

    monkeypatch.setattr(engine.hub, "resolve_dataset_content", timeout)
    card = cards.health_card("hf://org/data", fetch=True)
    assert card["status"] == "unknown"
    assert card["source_revision"]["kind"] == "unresolved"
    assert card["report"]["components"]["dataset_intake"]["status"] == "unknown"
    assert not any(f["status"] == "error" for f in card["report"]["findings"])
    assert 'fill="#276b37"' not in cards.render_badge(card)


@pytest.mark.parametrize("mutation", ["missing_component", "unsupported", "partial", "unresolved_revision"])
def test_false_pass_checker_cannot_promote_unassessed_card(tmp_path, monkeypatch, mutation):
    card = cards.health_card(str(dataset(tmp_path)))
    report = copy.deepcopy(card["report"])
    if mutation == "missing_component":
        del report["components"]["numeric_rows"]
    elif mutation == "unsupported":
        report["components"]["numeric_rows"]["status"] = "unsupported"
    elif mutation == "partial":
        report["components"]["numeric_rows"]["coverage"]["complete_parquet_scan"] = False
    else:
        report["artifacts"] = []
    monkeypatch.setattr(cards, "preflight", lambda *a, **k: report)
    changed = cards.health_card("local")
    assert changed["status"] == "unknown"
    assert 'fill="#276b37"' not in cards.render_badge(changed)


def test_batch_retains_each_result_continues_after_load_error_and_writes_index(tmp_path):
    root = dataset(tmp_path)
    result = cards.scan_health_cards([str(root), str(tmp_path / "missing"),
        {"source": str(root), "consumer": "other", "config": {"max_rows": 1}}], output_dir=tmp_path / "batch")
    assert [card["status"] for card in result["cards"]] == ["pass", "unknown", "unknown"]
    assert result["counts"] == {"error": 0, "unknown": 2, "warning": 0, "pass": 1}
    assert result["cards"][1]["report"]["intake_error"]["type"] == "InputError"
    index = json.loads((tmp_path / "batch/index.json").read_text())
    assert len(index["cards"]) == len(index["outputs"]) == 3
    assert all(cards.verify_health_card(card) for card in result["cards"])


@pytest.mark.parametrize("sources", [[], ["local"] * 17, ["local", {"source": "hf://org/data"}],
                                     ["local", {"source": "local", "fetch": True}], ["local", {}]])
def test_invalid_batch_structure_is_rejected_before_any_scan(monkeypatch, sources):
    monkeypatch.setattr(cards, "health_card", lambda *a, **k: pytest.fail("scan reached"))
    with pytest.raises(InputError):
        cards.scan_health_cards(sources)
