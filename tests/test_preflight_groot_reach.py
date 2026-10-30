"""Preflight, CLI and MCP wiring of the GR00T dataset contract and opt-in recorded-position residual triage."""

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_commanded_reach as reach_fixtures
import test_consumer_frontdoors as consumer_fixtures
import test_groot_contract as groot_fixtures
import test_preflight as preflight_fixtures
import test_readiness_workflow as readiness_fixtures

from policy_doctor import commanded_reach as reach_module
from policy_doctor import consumer_preflight, groot_contract, readiness_workflow
from policy_doctor import server as server_module
from policy_doctor.artifacts import InputError
from policy_doctor.cli import human, main
from policy_doctor.server import SUMMARY_LIMIT, _summary, create_server

engine = importlib.import_module("policy_doctor.preflight")
PASS_STATEMENT = "No residual pattern was found in the read rows."
N17_CONSUMER = "Isaac-GR00T N1.7 944fe70"
# Reuse the validated module fixtures: GR00T cube_to_bowl_5/SO100 metadata, residual rows, and a plain v3.0 dataset.
groot_dataset, groot_config, CONFIG = groot_fixtures._dataset, groot_fixtures._config, groot_fixtures.CONFIG
plain_dataset, hashes = preflight_fixtures.dataset, preflight_fixtures.hashes
# The real ordinary GR00T consumer-comparison job (its dataset carries meta/modality.json), registered here.
consumer_job, ordinary = consumer_fixtures.consumer_job, consumer_fixtures.ordinary
SEVEN_DOF = groot_fixtures._modality(state__single_arm={"start": 0, "end": 6}, state__gripper={"start": 6, "end": 7},
                                     action__single_arm={"start": 0, "end": 6}, action__gripper={"start": 6, "end": 7})


@pytest.fixture
def reach(monkeypatch):
    # The residual fixtures draw from one module-level generator; give these tests their own.
    monkeypatch.setattr(reach_fixtures, "RNG", np.random.default_rng(7))
    return reach_fixtures


def combined(tmp_path, reach, modality=groot_fixtures.MODALITY, **kwargs):
    """The GR00T contract fixture metadata plus the residual fixture rows (same six `<joint>.pos` names)."""
    root = groot_dataset(tmp_path / "groot", modality)
    rows = reach._dataset(tmp_path / "rows", **kwargs) / "data" / "chunk-000" / "file-000.parquet"
    (root / "data" / "chunk-000").mkdir(parents=True)
    shutil.copyfile(rows, root / "data" / "chunk-000" / "episode_000000.parquet")
    return root


def lerobot_rows(tmp_path, fixtures, name="lerobot", **kwargs):
    """The residual fixture rows in a complete LeRobot v3.0 layout on which every existing preflight check passes."""
    source = fixtures._dataset(tmp_path / (name + "_source"), **kwargs) / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(source).to_pydict()
    episodes = table["episode_index"]
    starts = {e: episodes.index(e) for e in set(episodes)}
    frames = [i - starts[e] for i, e in enumerate(episodes)]
    root = tmp_path / name
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    feature = {"dtype": "float32", "shape": [len(fixtures.JOINTS)], "names": fixtures.JOINTS}
    info = {"codebase_version": "v3.0", "fps": 30, "total_frames": len(episodes), "total_episodes": len(starts),
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {"action": feature, "observation.state": dict(feature)}}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    stats = {}
    for key in ("action", "observation.state"):
        values = np.asarray(table[key])
        stats[key] = {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                      "min": values.min(0).tolist(), "max": values.max(0).tolist()}
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    pq.write_table(pa.Table.from_pydict({"index": list(range(len(episodes))), "episode_index": episodes,
                                         "frame_index": frames, "timestamp": [f / 30 for f in frames],
                                         "action": table["action"], "observation.state": table["observation.state"]}),
                   root / "data" / "chunk-000" / "file-000.parquet")
    ends = {e: len(episodes) - episodes[::-1].index(e) for e in starts}
    pq.write_table(pa.Table.from_pylist([{"episode_index": e, "length": ends[e] - starts[e],
                                          "dataset_from_index": starts[e], "dataset_to_index": ends[e],
                                          "data/chunk_index": 0, "data/file_index": 0} for e in sorted(starts)]),
                   root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root


def structured(result):
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and isinstance(result[-1], dict):
        return result[-1]
    return json.loads(result[0].text)


def forbidden(*args, **kwargs):
    raise AssertionError("this reader must not be reached")


def hub_intake(root, calls):
    @contextmanager
    def intake(source, **kwargs):
        calls.append(source)
        yield SimpleNamespace(root=root, provenance={"source": source, "resolved_revision": "a" * 40,
            "files": [{"path": k, "sha256": v} for k, v in hashes(root).items()]}, video_context=None)
    return intake


# ---------------------------------------------------------------- GR00T dataset contract

def test_groot_layout_runs_the_contract_automatically_and_names_the_missing_config(tmp_path):
    root = groot_dataset(tmp_path / "ds")
    result = engine.preflight(str(root))
    part = result["components"]["groot_contract"]
    assert part["kind"] == "groot_dataset_contract" and part["title"] == "GR00T dataset contract"
    assert part["status"] == "error" and part["versions_run"] == ["n1.5", "n1.7"]
    assert part["schema_version"] == 3 and part["mode"] == "both" and part["fails_on"] == ["n1.5"]
    assert part["versions"]["n1.5"]["status"] == "error" and part["versions"]["n1.7"]["status"] == "unknown"
    assert part["versions"]["n1.7"]["consumer"] == N17_CONSUMER
    assert any(f["code"] == "groot.video_column" and f["status"] == "error" and f["groot_version"] == "n1.5"
               for f in part["findings"])  # fixture lacks N1.5's required video-channel metadata
    assert any(f["code"] == "groot.config" and f["status"] == "unknown" and f["component"] == "groot_contract"
               for f in result["findings"])
    assert part["coverage"]["modality_config_supplied"] is False and not part["coverage"]["complete"]
    assert part["coverage"]["parquet_rows_read"] is False and part["input_sources"] == [str(root)]
    assert result["coverage"]["groot_contract_requested"] is False
    assert result["coverage"]["components"]["groot_contract"] == "error"
    assert any("GR00T training modality config" in action for action in result["next_actions"])
    assert "commanded_reach" not in result["components"]
    modality = next(a for a in part["artifacts"] if a["path"].endswith("meta/modality.json"))
    assert modality["sha256"] == hashlib.sha256((root / "meta/modality.json").read_bytes()).hexdigest()
    assert "never executed" in result["scope"] and "named version profiles" in result["scope"]


def test_plain_lerobot_dataset_is_unchanged(tmp_path):
    result = engine.preflight(str(plain_dataset(tmp_path)))
    assert result["status"] == "pass"
    assert not {"groot_contract", "commanded_reach"} & result["components"].keys()
    assert "groot_contract_requested" not in result["coverage"] and "GR00T" not in result["scope"]


def test_supplied_config_is_checked_and_retained(tmp_path):
    root, config = groot_dataset(tmp_path / "ds"), groot_config(tmp_path)
    result = engine.preflight(str(root), groot_modality_config=str(config))
    part = result["components"]["groot_contract"]
    assert part["status"] == "pass" and part["modality_config_name"] == "so100_config"
    assert part["coverage"]["complete"] and part["coverage"]["modality_config_supplied"]
    assert part["input_sources"] == [str(root), str(config.resolve())]
    assert result["coverage"]["groot_contract_requested"] is True
    identity = next(a for a in part["artifacts"] if a["role"] == "groot_modality_config")
    assert identity["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    # Both readers of meta/info.json saw the same bytes.
    assert "input_stability" not in result["components"]
    info = [a for c in result["components"].values() for a in c.get("artifacts", [])
            if a["path"] == str((root / "meta/info.json").resolve())]
    assert len(info) >= 2 and len({a["sha256"] for a in info}) == 1


def test_silent_groot_slice_fault_is_a_preflight_error(tmp_path):
    seven = groot_fixtures._modality(state__single_arm={"start": 0, "end": 6}, state__gripper={"start": 6, "end": 7},
                                     action__single_arm={"start": 0, "end": 6}, action__gripper={"start": 6, "end": 7})
    result = engine.preflight(str(groot_dataset(tmp_path / "ds", seven)), groot_modality_config=str(groot_config(tmp_path)))
    assert result["components"]["groot_contract"]["status"] == result["status"] == "error"
    errors = {f["code"] for f in result["findings"] if f["status"] == "error" and f["component"] == "groot_contract"}
    assert {"groot.state_slice", "groot.action_slice"} <= errors
    assert all(f["applies_to"] for f in result["findings"] if f["component"] == "groot_contract")


def test_config_name_selects_one_dict_and_is_required_where_ambiguous(tmp_path):
    body = CONFIG.split("so100_config = ", 1)[1].rsplit("\nregister_modality_config", 1)[0]
    text = ("from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig\n"
            f"MODALITY_CONFIGS = {{\n\"so100\": {body},\n\"other\": {body.replace('front', 'top')},\n}}\n")
    path, root = groot_config(tmp_path, text, "embodiment_configs.py"), groot_dataset(tmp_path / "ds")
    named = engine.preflight(str(root), groot_modality_config=str(path), groot_config_name="so100")
    assert named["components"]["groot_contract"]["status"] == "pass"
    assert named["components"]["groot_contract"]["modality_config_name"] == "so100"
    ambiguous = engine.preflight(str(root), groot_modality_config=str(path))["components"]["groot_contract"]
    assert ambiguous["status"] == "unknown" and "config_name" in ambiguous["findings"][0]["message"]
    missing = engine.preflight(str(root), groot_modality_config=str(path), groot_config_name="absent")
    assert "named 'absent'" in missing["components"]["groot_contract"]["findings"][0]["message"]


def test_relative_config_resolves_against_the_base_directory(tmp_path):
    root, config = groot_dataset(tmp_path / "ds"), groot_config(tmp_path / "ds")
    result = engine.preflight(str(root), groot_modality_config=config.name, base_directory=str(root))
    assert result["components"]["groot_contract"]["status"] == "pass"


def test_hub_source_reports_requested_components_unknown_without_reading_the_config(tmp_path, monkeypatch):
    root, config, calls = groot_dataset(tmp_path / "ds"), groot_config(tmp_path), []
    monkeypatch.setattr(engine.hub, "resolve_dataset_content", hub_intake(root, calls))
    monkeypatch.setattr(groot_contract, "check_groot_dataset_versions", forbidden)
    monkeypatch.setattr(reach_module, "check_commanded_reach", forbidden)
    monkeypatch.setattr(engine, "_config_identity", forbidden)
    result = engine.preflight("hf://org/data", fetch=True, revision="a" * 40, groot_modality_config=str(config),
                              commanded_reach=True)
    assert calls == ["hf://org/data"]
    for name in ("groot_contract", "commanded_reach"):
        part = result["components"][name]
        assert part["status"] == "unknown" and "requires a local dataset" in part["findings"][0]["message"]
    assert result["components"]["commanded_reach"]["residual_statement"] != PASS_STATEMENT
    # Without GR00T parameters a Hub source is not given the contract, even with meta/modality.json present.
    plain = engine.preflight("hf://org/data", fetch=True, revision="a" * 40)
    assert not {"groot_contract", "commanded_reach"} & plain["components"].keys()


def test_config_outside_read_roots_is_refused_before_any_intake(tmp_path, monkeypatch):
    roots = tmp_path / "roots"
    root, outside = groot_dataset(roots / "ds"), groot_config(tmp_path)
    link = roots / "link.py"
    link.symlink_to(outside)
    monkeypatch.setattr(groot_contract, "check_groot_dataset_versions", forbidden)
    monkeypatch.setattr(engine.hub, "resolve_dataset_content", forbidden)
    for source in (str(root), "hf://org/data"):
        for value in (str(outside), str(link)):
            with pytest.raises(InputError, match="outside the configured read roots"):
                engine.preflight(source, fetch=True, groot_modality_config=value, read_roots=[str(roots)])


def test_contract_never_opens_metadata_outside_read_roots(tmp_path, monkeypatch):
    # A Hugging Face cache snapshot links its files into blobs/; the contract accepts that layout on its own.
    source = groot_dataset(tmp_path / "plain")
    cache = tmp_path / "datasets--org--name"
    (cache / "blobs").mkdir(parents=True)
    snapshot = cache / "snapshots" / "abc123"
    (snapshot / "meta").mkdir(parents=True)
    for path in (source / "meta").iterdir():
        blob = cache / "blobs" / hashlib.sha256(path.read_bytes()).hexdigest()
        blob.write_bytes(path.read_bytes())
        os.symlink(os.path.relpath(blob, snapshot / "meta"), snapshot / "meta" / path.name)
    config = groot_config(snapshot)
    assert engine.preflight(str(snapshot), groot_modality_config=str(config))["components"]["groot_contract"]["status"] == "pass"
    monkeypatch.setattr(groot_contract, "check_groot_dataset_versions", forbidden)
    part = engine.preflight(str(snapshot), groot_modality_config=str(config),
                            read_roots=[str(snapshot)])["components"]["groot_contract"]
    assert part["status"] == "unknown" and "outside the configured read roots" in part["findings"][0]["message"]


@pytest.mark.parametrize("kwargs", [{"commanded_reach": 1}, {"commanded_reach": "yes"}, {"commanded_reach": None},
                                    {"groot_config_name": "so100", "groot_version": "n1.7"}, {"groot_config_name": ""},
                                    {"groot_config_name": 3, "groot_modality_config": "c.py"},
                                    {"groot_config_name": "x" * 257, "groot_modality_config": "c.py"},
                                    {"groot_modality_config": ""}, {"groot_modality_config": "hf://org/config"},
                                    {"groot_modality_config": 1}])
def test_invalid_new_arguments_are_refused_before_checks(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(engine, "scan_dataset", forbidden)
    with pytest.raises(InputError):
        engine.preflight(str(tmp_path), **kwargs)


# ---------------------------------------------------------------- recorded-position residual triage

def test_residual_triage_is_opt_in_and_a_pass_claims_only_the_read_rows(tmp_path, reach):
    root = reach._dataset(tmp_path / "ds")
    assert "commanded_reach" not in engine.preflight(str(root))["components"]
    result = engine.preflight(str(root), commanded_reach=True)
    part = result["components"]["commanded_reach"]
    assert part["status"] == "pass" and part["title"] == "Recorded-position residual triage"
    assert part["residual_statement"] == PASS_STATEMENT
    assert part["coverage"]["rows"] > 1000 and part["coverage"]["data_files_read"] == 1
    assert part["coverage"]["max_rows"] == 200000 and "data_files" not in part["coverage"]
    assert result["coverage"]["commanded_reach_requested"] is True
    # The read Parquet file is recorded once with its hash, here or by another component that hashed it too.
    path = str((root / "data/chunk-000/file-000.parquet").resolve())
    recorded = [a for c in result["components"].values() for a in c.get("artifacts", []) if a["path"] == path]
    assert recorded and {a["sha256"] for a in recorded} == {hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    assert len([a for a in part["artifacts"] if a["path"] == path]) + \
        part["coverage"]["files_with_matching_hash_in_other_components"] >= 1
    assert "neither a residual nor its absence establishes calibration or robot health" in result["scope"]


def test_residual_warning_is_triage_evidence(tmp_path, reach):
    result = engine.preflight(str(reach._dataset(tmp_path / "ds", reach=reach._clip(4, high=150.0))),
                              commanded_reach=True)
    part = result["components"]["commanded_reach"]
    assert part["status"] == "warning" and "residual pattern" in part["residual_statement"]
    finding = next(f for f in result["findings"] if f["code"] == "reach.held_at_an_end")
    assert finding["component"] == "commanded_reach" and finding["evidence"][0]["value"] == "wrist_roll.pos"
    assert any("what the motors received is not recorded" in action for action in result["next_actions"])


def test_wiring_text_never_implies_calibration_delivery_or_health(tmp_path, reach):
    from policy_doctor.cli import parser

    result = engine.preflight(str(reach._dataset(tmp_path / "ds", reach=reach._clip(4, high=150.0))),
                              commanded_reach=True)
    commands = next(a for a in parser()._actions if isinstance(a, argparse._SubParsersAction))
    helps = [a.help for a in commands.choices["preflight"]._actions
             if a.dest in ("commanded_reach", "groot_config", "groot_config_name")]
    tools = asyncio.run(create_server([str(tmp_path)]).list_tools())
    description = next(t.description for t in tools if t.name == "preflight_check")
    added = description.split("commanded_reach=true", 1)[1].split(" To compare", 1)[0]
    texts = [*engine.TITLES.values(), *engine.RESIDUAL_STATEMENTS.values(), engine.RESIDUAL_LIMITS,
             *[a for a in result["next_actions"] if "joints" in a], result["scope"].split("Opt-in", 1)[1],
             *helps, added]
    assert len(helps) == 3 and "robot-health" in helps[-1]
    for text in texts:
        for sentence in re.split(r"(?<=[.;])\s+", " ".join(text.split())):
            if re.search(r"calibrat|health|actuator|deliver|received", sentence, re.I):
                assert re.search(r"\b(not|neither|nor|no)\b", sentence, re.I), sentence
            assert not re.search(r"\b(healthy|calibrated|certif\w*|verified|reached)\b", sentence, re.I), sentence


def test_residual_unknowns_and_input_stability(tmp_path, monkeypatch):
    root = plain_dataset(tmp_path)  # action names a/b are not `<joint>.pos` motor positions
    result = engine.preflight(str(root), commanded_reach=True)
    part = result["components"]["commanded_reach"]
    assert part["status"] == "unknown" and part["findings"][0]["code"] == "reach.not_applicable"
    assert part["residual_statement"] == engine.RESIDUAL_STATEMENTS["unknown"]
    assert "input_stability" not in result["components"]
    original = reach_module.check_commanded_reach

    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        report["findings"][0]["evidence"][0]["sha256"] = "0" * 64
        return report

    monkeypatch.setattr(reach_module, "check_commanded_reach", changed)
    unstable = engine.preflight(str(root), commanded_reach=True)
    assert unstable["components"]["input_stability"]["status"] == unstable["status"] == "unknown"


def test_more_rows_than_the_budget_stay_unknown(tmp_path, reach):
    part = engine.preflight(str(reach._dataset(tmp_path / "ds")), commanded_reach=True,
                            max_rows=100)["components"]["commanded_reach"]
    assert part["status"] == "unknown" and "raise max_rows" in part["findings"][0]["message"]
    assert part["residual_statement"] != PASS_STATEMENT


def test_residual_triage_respects_read_roots(tmp_path):
    root = plain_dataset(tmp_path)
    part = engine.preflight(str(root), commanded_reach=True, read_roots=[str(tmp_path / "elsewhere")])
    message = part["components"]["commanded_reach"]["findings"][0]["message"]
    assert "outside the configured read roots" in message


# ---------------------------------------------------------------- co-running with existing components

def test_new_components_co_run_with_checkpoint_comparison(tmp_path, reach, case):
    root, config = combined(tmp_path, reach), groot_config(tmp_path)
    result = engine.preflight(str(root), checkpoint=str(case["checkpoint"]), groot_modality_config=str(config),
                              commanded_reach=True)
    assert {"checkpoint_metadata", "groot_contract", "commanded_reach"} <= result["components"].keys()
    assert result["components"]["groot_contract"]["status"] == "pass"
    assert result["components"]["commanded_reach"]["status"] == "pass"
    assert "input_stability" not in result["components"]


def test_new_components_co_run_with_consumer_comparison_or_robot_readiness_not_both(tmp_path, reach, monkeypatch):
    root, config = combined(tmp_path, reach), groot_config(tmp_path)
    plan = {"reference_dataset": str(root), "reference_checkpoint": str(tmp_path / "ref"), "samples": None,
            "require_equal": False}
    monkeypatch.setattr(consumer_preflight, "prepare_consumer_comparison",
                        lambda *a, **k: plan if k.get("reference_checkpoint") else None)
    monkeypatch.setattr(consumer_preflight, "check_consumer_comparison",
                        lambda *a, **k: {"status": "pass", "findings": [], "artifacts": []})
    monkeypatch.setattr(readiness_workflow, "prepare_readiness",
                        lambda **k: {"checkpoint": None, "policy_conflict": False})
    monkeypatch.setattr(readiness_workflow, "check_robot_readiness",
                        lambda **k: {"status": "pass", "findings": [], "axes": {}, "provenance": []})
    options = {"groot_modality_config": str(config), "commanded_reach": True}
    consumer = engine.preflight(str(root), checkpoint=str(tmp_path / "cand"),
                                consumer_reference_checkpoint=str(tmp_path / "ref"), **options)
    assert {"consumer_comparison", "groot_contract", "commanded_reach"} <= consumer["components"].keys()
    robot = engine.preflight(str(root), robot_command="lerobot-record --robot.type=so101_follower", **options)
    assert {"robot_readiness", "groot_contract", "commanded_reach"} <= robot["components"].keys()
    assert robot["components"]["commanded_reach"]["status"] == "pass"
    with pytest.raises(InputError, match="separate preflight calls"):
        engine.preflight(str(root), checkpoint=str(tmp_path / "cand"), consumer_reference_checkpoint=str(tmp_path / "ref"),
                         robot_command="lerobot-record --robot.type=so101_follower", **options)


# ---------------------------------------------------------------- CLI and rendering

def test_cli_flags_reach_the_engine_and_render_in_every_format(tmp_path, reach, capsys, monkeypatch):
    root, config = combined(tmp_path, reach), groot_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    arguments = ["preflight", str(root), "--groot-version", "n1.7", "--groot-config", config.name, "--commanded-reach"]
    main(arguments + ["--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["components"]["groot_contract"]["status"] == "pass"
    assert report["components"]["groot_contract"]["input_sources"] == [str(root), str(config.resolve())]
    assert report["components"]["commanded_reach"]["residual_statement"] == PASS_STATEMENT
    main(arguments)
    text = capsys.readouterr().out
    assert "  GR00T dataset contract: pass" in text and "  Recorded-position residual triage: pass" in text
    assert "    consumer: Isaac-GR00T N1.7 944fe70; modality config: so100_config" in text
    assert "    " + PASS_STATEMENT in text and "    " + engine.RESIDUAL_LIMITS in text
    assert "read rows: " in text and "commanded reach" not in text
    main(arguments + ["--markdown"])
    markdown = capsys.readouterr().out
    assert markdown.startswith("# RoboClue dataset preflight")
    assert "Recorded-position residual triage: pass" in markdown and "so100\\_config" in markdown


def test_cli_defaults_and_refusals(tmp_path, capsys, monkeypatch):
    from policy_doctor.cli import parser

    args = parser().parse_args(["preflight", "ds"])
    assert args.commanded_reach is False and args.groot_config is None and args.groot_config_name is None
    assert args.groot_version is None
    observed = []
    monkeypatch.setattr(engine, "preflight", lambda source, **k: observed.append(k) or {"status": "pass", "findings": []})
    assert main(["preflight", "ds", "--groot-config", "c.py", "--groot-config-name", "so100", "--json"]) == 0
    assert observed[-1]["groot_modality_config"] == "c.py" and observed[-1]["groot_config_name"] == "so100"
    assert observed[-1]["commanded_reach"] is False
    assert observed[-1]["groot_version"] is None
    capsys.readouterr()
    monkeypatch.undo()
    root = groot_dataset(tmp_path / "ds")
    assert engine.preflight(str(root), groot_config_name="so100")["components"]["groot_contract"]["versions_run"] == ["n1.5"]
    assert main(["preflight", str(root), "--groot-version", "n1.7", "--groot-config-name", "so100"]) == 2
    assert "requires --groot-config" in json.loads(capsys.readouterr().err)["error"]["message"]


def only(result, name):
    """Render one component without the unrelated components' findings competing for the 12 shown lines."""
    return dict(result, components={name: result["components"][name]},
                findings=[f for f in result["findings"] if f["component"] == name])


def test_human_rendering_of_unknown_and_warning_components(tmp_path, reach):
    root = reach._dataset(tmp_path / "ds", reach=reach._clip(4, high=150.0))
    text = human(only(engine.preflight(str(root), commanded_reach=True), "commanded_reach"))
    assert "  Recorded-position residual triage: warning" in text
    assert "    " + engine.RESIDUAL_STATEMENTS["warning"] in text
    assert "[WARNING] reach.held_at_an_end: wrist_roll.pos" in text
    unknown = human(only(engine.preflight(str(groot_dataset(tmp_path / "g")), groot_version="n1.7"), "groot_contract"))
    assert "  GR00T dataset contract: unknown" in unknown and "modality config: not supplied" in unknown
    assert "[UNKNOWN] groot.config:" in unknown and "(applies to fine-tuning and direct loaders)" in unknown


# ---------------------------------------------------------------- MCP

def test_mcp_parameters_are_typed_and_the_default_tool_count_is_unchanged(tmp_path):
    tools = asyncio.run(create_server([str(tmp_path)]).list_tools())
    assert len(tools) == 29
    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in tools}
    schema = next(t for t in tools if t.name == "preflight_check").inputSchema
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert properties["commanded_reach"]["type"] == "boolean"
    for name in ("groot_modality_config", "groot_config_name"):
        assert {"type": "string"} in properties[name]["anyOf"] and {"type": "null"} in properties[name]["anyOf"]
    assert {"type": "string", "enum": ["n1.5", "n1.7"]} in properties["groot_version"]["anyOf"]
    assert {"type": "null"} in properties["groot_version"]["anyOf"] and properties["groot_version"]["default"] is None
    assert not {"commanded_reach", "groot_modality_config", "groot_config_name", "groot_version"} & set(schema.get("required", []))


def test_mcp_authorizes_the_config_before_preflight(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    root, outside, inside = groot_dataset(allowed / "ds"), groot_config(tmp_path), groot_config(allowed)
    (allowed / "link.py").symlink_to(outside)
    observed = []
    monkeypatch.setattr(engine, "preflight", lambda source, **k: observed.append((source, k)) or
                        {"kind": "dataset_preflight", "status": "pass", "findings": []})
    server = create_server([str(allowed)])

    async def run():
        for value, expected in ((str(outside), "outside"), (str(allowed / "link.py"), "outside"),
                                ("hf://org/config", "local"), ("", "nonempty")):
            with pytest.raises(Exception, match=expected):
                await server.call_tool("preflight_check", {"source": str(root), "groot_modality_config": value})
        assert observed == []
        await server.call_tool("preflight_check", {"source": str(root), "groot_modality_config": str(inside),
                                                   "groot_config_name": "so100_config", "commanded_reach": True})
    asyncio.run(run())
    source, kwargs = observed[0]
    assert source == str(root.resolve()) and kwargs["groot_modality_config"] == str(inside.resolve())
    assert kwargs["groot_config_name"] == "so100_config" and kwargs["commanded_reach"] is True
    assert kwargs["read_roots"] == [str(allowed.resolve())]


def test_mcp_strict_boolean_and_concise_component_summaries(tmp_path, reach):
    root, config = combined(tmp_path, reach), groot_config(tmp_path)
    server = create_server([str(tmp_path)])

    async def run():
        with pytest.raises(Exception, match="strict boolean"):
            await server.call_tool("preflight_check", {"source": str(root), "commanded_reach": "yes"})
        return structured(await server.call_tool("preflight_check", {
            "source": str(root), "groot_modality_config": str(config), "commanded_reach": True}))
    concise = asyncio.run(run())
    assert len(json.dumps(concise).encode()) <= SUMMARY_LIMIT
    groot = concise["components"]["groot_contract"]
    assert groot["status"] == "pass" and groot["consumer"] == N17_CONSUMER
    assert groot["modality_config_supplied"] is True and groot["modality_config_name"] == "so100_config"
    reach_part = concise["components"]["commanded_reach"]
    assert reach_part["status"] == "pass" and reach_part["residual_statement"] == PASS_STATEMENT
    assert reach_part["limits"] == engine.RESIDUAL_LIMITS and reach_part["rows"] > 1000


def test_mcp_detailed_response_respects_the_size_bound(tmp_path, reach, monkeypatch):
    root, config = combined(tmp_path, reach), groot_config(tmp_path)
    server = create_server([str(tmp_path)])
    arguments = {"source": str(root), "groot_modality_config": str(config), "commanded_reach": True,
                 "response_format": "detailed"}
    raw = engine.preflight(str(root), groot_modality_config=str(config), commanded_reach=True)
    assert raw["components"]["commanded_reach"]["joints"]
    # Exercise both response branches without assuming a temporary-path length.
    monkeypatch.setattr(server_module, "RESULT_LIMIT",
                        max(server_module.RESULT_LIMIT, len(json.dumps(raw).encode()) + 4096))
    full = structured(asyncio.run(server.call_tool("preflight_check", arguments)))
    assert full["kind"] == "dataset_preflight" and len(json.dumps(full).encode()) <= server_module.RESULT_LIMIT
    assert "truncated_from_bytes" not in full
    assert full["components"]["commanded_reach"]["joints"]
    monkeypatch.setattr(server_module, "RESULT_LIMIT", 4096)
    bounded = structured(asyncio.run(server.call_tool("preflight_check", arguments)))
    assert bounded["truncated_from_bytes"] > 4096 and bounded["status"] == full["status"]
    assert bounded["components"]["commanded_reach"]["residual_statement"] == PASS_STATEMENT
    assert bounded["components"]["groot_contract"]["status"] == "pass"


def test_summary_keeps_every_component_within_its_bound():
    long = "x" * 2000
    components = {f"component_{i}": {"status": "pass"} for i in range(10)}
    components["groot_contract"] = {"status": "error", "title": "GR00T dataset contract", "consumer": "Isaac-GR00T 944fe70",
                                    "counts": {"error": 40}, "modality_config_name": long,
                                    "coverage": {"modality_config_supplied": True}}
    components["commanded_reach"] = {"status": "warning", "title": "Recorded-position residual triage",
                                     "residual_statement": engine.RESIDUAL_STATEMENTS["warning"],
                                     "residual_limits": engine.RESIDUAL_LIMITS, "counts": {"warning": 6},
                                     "coverage": {"rows": 200000, "episodes": 50}}
    report = {"kind": "dataset_preflight", "status": "error", "scope": long, "components": components,
              "findings": [{"code": "groot.state_slice", "status": "error", "message": long, "remedy": long,
                            "evidence": [{"path": long, "pointer": "/state", "sha256": "a" * 64, "value": long}]}] * 80,
              "next_actions": [long] * 8}
    summary = _summary(report)
    assert len(json.dumps(summary).encode()) <= SUMMARY_LIMIT
    assert set(components) <= set(summary["components"])
    assert summary["components"]["commanded_reach"]["limits"] == engine.RESIDUAL_LIMITS
    assert len(summary["components"]["groot_contract"]["modality_config_name"]) == 120


# ---------------------------------------------------------------- coordinator requirements (0.40 wiring review)
# R1 coexistence with the real consumer comparison and the real robot readiness, through the SDK and the CLI.

def overall(result):
    return next(s for s in engine.PRIORITY if s in {c["status"] for c in result["components"].values()})


def test_r1_real_consumer_comparison_co_runs_through_sdk_and_cli(request, tmp_path, capsys):
    job = request.getfixturevalue("consumer_job")
    result = engine.preflight(**consumer_fixtures.request(job, commanded_reach=True, groot_version="n1.7"))
    parts = result["components"]
    assert parts["consumer_comparison"]["status"] == "pass"
    assert parts["consumer_comparison"]["numeric_projection_value_effect"] == "CHANGED"
    # The ordinary G1 dataset carries meta/modality.json, so the contract runs without a config and says so.
    assert parts["groot_contract"]["kind"] == "groot_dataset_contract"
    assert parts["groot_contract"]["coverage"]["modality_config_supplied"] is False
    assert parts["groot_contract"]["status"] != "pass"
    # Its 43-wide unnamed vectors are not `<joint>.pos` positions: an explicit unknown, not a skip.
    assert parts["commanded_reach"]["status"] == "unknown"
    assert parts["commanded_reach"]["findings"][0]["code"] == "reach.not_applicable"
    assert result["status"] == overall(result) and result["status"] in ("unknown", "error")
    concise = _summary(result)
    consumer_fixtures.assert_source_witness(concise["components"]["consumer_comparison"],
                                            consumer_fixtures.parquet_identity(job))
    assert concise["components"]["commanded_reach"]["limits"] == engine.RESIDUAL_LIMITS
    assert len(json.dumps(concise, allow_nan=False).encode()) <= SUMMARY_LIMIT
    strict = engine.preflight(**consumer_fixtures.request(job, commanded_reach=True, consumer_require_equal=True,
                                                        groot_version="n1.7"))
    assert strict["components"]["consumer_comparison"]["status"] == strict["status"] == "error"
    code = main(consumer_fixtures.cli_args(job) + ["--groot-version", "n1.7", "--commanded-reach", "--json"])
    cli = json.loads(capsys.readouterr().out)
    assert code == {"error": 1, "unknown": 3}[cli["status"]] and cli["status"] == result["status"]
    assert {name: part["status"] for name, part in cli["components"].items()} == \
        {name: part["status"] for name, part in parts.items()}
    main(consumer_fixtures.cli_args(job) + ["--groot-version", "n1.7", "--commanded-reach"])
    text = capsys.readouterr().out
    assert "consumer comparison: pass" in text and "selected numeric values: CHANGED" in text
    assert "Recorded-position residual triage: unknown" in text and "GR00T dataset contract:" in text


def test_r1_consumer_comparison_and_robot_readiness_still_refuse_together(request, tmp_path, capsys):
    job = request.getfixturevalue("consumer_job")
    robot = str(tmp_path / "robot.json")  # refused before it is read
    with pytest.raises(InputError, match="separate preflight calls"):
        engine.preflight(**consumer_fixtures.request(job, robot_config=robot, commanded_reach=True,
                                                     groot_modality_config=str(groot_config(tmp_path))))
    assert main(consumer_fixtures.cli_args(job) + ["--robot-config", robot, "--commanded-reach"]) == 2
    assert "separate preflight calls" in json.loads(capsys.readouterr().err)["error"]["message"]


def test_r1_real_robot_readiness_co_runs_through_sdk_and_cli(case, reach, tmp_path, capsys):
    path = readiness_fixtures.healthy(case)  # so100_follower: the same six `<joint>.pos` names as the rows
    rows = reach._dataset(tmp_path / "rows") / "data" / "chunk-000" / "file-000.parquet"
    (case["dataset"] / "data" / "chunk-000").mkdir(parents=True)
    shutil.copyfile(rows, case["dataset"] / "data" / "chunk-000" / "file-000.parquet")
    config = groot_config(tmp_path)
    result = engine.preflight(str(case["dataset"]), robot_config=str(path), video_headers=False,
                              commanded_reach=True, groot_modality_config=str(config))
    parts = result["components"]
    assert parts["robot_readiness"]["status"] == "pass" and len(parts["robot_readiness"]["axes"]) == 10
    assert parts["commanded_reach"]["status"] == "pass"
    assert parts["commanded_reach"]["residual_statement"] == PASS_STATEMENT
    # A LeRobot v3.0 dataset without meta/modality.json is not what GR00T 944fe70 reads: a known error dominates.
    assert parts["groot_contract"]["status"] == result["status"] == "error"
    assert "groot.modality" in {f["code"] for f in parts["groot_contract"]["findings"]}
    code = main(["preflight", str(case["dataset"]), "--robot-config", str(path), "--no-video-headers",
                 "--commanded-reach", "--groot-config", str(config), "--json"])
    cli = json.loads(capsys.readouterr().out)
    assert code == 1 and cli["status"] == "error"
    assert {name: part["status"] for name, part in cli["components"].items()} == \
        {name: part["status"] for name, part in parts.items()}
    concise = _summary(result)
    assert concise["components"]["robot_readiness"]["axes"] and concise["components"]["commanded_reach"]["status"] == "pass"


# R2 overall status and counts when a new component reports each status.

def synthetic(kind, status):
    report = {"kind": kind, "status": status, "counts": {s: int(s == status) for s in ("error", "warning", "unknown", "pass")},
              "findings": [{"code": kind + ".generated", "status": status, "message": "generated " + status,
                            "evidence": []}]}
    if kind == "groot_dataset_contract":
        report.update(schema_version=3, mode="explicit", files={}, modality_config_name="generated",
                      consumer=N17_CONSUMER, coverage={"not_checked": [], "modality_config_supplied": True},
                      groot_version="n1.7", versions_run=["n1.7"], fails_on=["n1.7"] if status == "error" else [],
                      headline="Synthetic N1.7 " + status, remedy="", total_findings=1, findings_truncated=False,
                      finding_codes={kind + ".generated": {status: 1}},
                      versions={"n1.7": {"consumer": N17_CONSUMER, "status": status, "counts": report["counts"],
                                         "applies_to": [], "statement": "Synthetic N1.7 fixture", "not_checked": []}})
        report["findings"][0]["groot_version"] = "n1.7"
    else:
        report.update(coverage={"rows": 20, "episodes": 2, "data_files": []}, joints={})
    return report


@pytest.mark.parametrize("component", ["groot_contract", "commanded_reach"])
@pytest.mark.parametrize("status", ["pass", "warning", "unknown", "error"])
def test_r2_new_component_status_and_counts_are_preserved(tmp_path, monkeypatch, capsys, component, status):
    root = plain_dataset(tmp_path)  # every existing component passes on it
    baseline = engine.preflight(str(root))
    assert baseline["status"] == "pass"
    if component == "groot_contract":
        monkeypatch.setattr(groot_contract, "check_groot_dataset_versions",
                            lambda *a, **k: synthetic("groot_dataset_contract", status))
        flags = ["--groot-config", str(groot_config(tmp_path)), "--groot-version", "n1.7"]
        options = {"groot_modality_config": flags[1], "groot_version": "n1.7"}
    else:
        monkeypatch.setattr(reach_module, "check_commanded_reach", lambda *a, **k: synthetic("commanded_reach", status))
        flags, options = ["--commanded-reach"], {"commanded_reach": True}
    result = engine.preflight(str(root), **options)
    assert result["components"][component]["status"] == result["coverage"]["components"][component] == status
    assert result["status"] == status
    assert result["counts"] == {s: baseline["counts"][s] + (s == status) for s in engine.PRIORITY}
    assert result["total_findings"] == baseline["total_findings"] + 1
    concise = _summary(result)
    assert concise["status"] == status and concise["counts"] == result["counts"]
    assert concise["components"][component]["status"] == status
    assert main(["preflight", str(root), "--json"] + flags) == {"error": 1, "unknown": 3}.get(status, 0)
    assert json.loads(capsys.readouterr().out)["counts"] == result["counts"]


def test_r2_real_contract_error_dominates_passing_and_unknown_components(tmp_path, capsys):
    root = plain_dataset(tmp_path)
    (root / "meta" / "modality.json").write_text(json.dumps(groot_fixtures.MODALITY))
    result = engine.preflight(str(root), commanded_reach=True)
    parts = result["components"]
    assert all(parts[name]["status"] == "pass" for name in ("dataset_statistics", "numeric_rows", "episode_windows"))
    assert parts["commanded_reach"]["status"] == "unknown"  # a/b are not motor positions
    groot_errors = [f for f in parts["groot_contract"]["findings"] if f["status"] == "error"]
    assert groot_errors and "groot.format" in {f["code"] for f in groot_errors}  # v3.0, not LeRobot v2
    assert parts["groot_contract"]["status"] == result["status"] == "error"
    assert result["counts"]["error"] == sum(f["status"] == "error" for f in result["findings"]) == len(groot_errors)
    assert result["counts"]["unknown"] == sum(f["status"] == "unknown" for f in result["findings"]) >= 2
    assert main(["preflight", str(root), "--commanded-reach", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["counts"] == result["counts"]


def test_r2_real_residual_warning_does_not_hide_unknown_components(tmp_path, reach):
    result = engine.preflight(str(reach._dataset(tmp_path / "ds", reach=reach._clip(4, high=150.0))),
                              commanded_reach=True)
    assert result["components"]["commanded_reach"]["status"] == "warning"
    assert result["status"] == overall(result)
    assert result["counts"]["warning"] == sum(f["status"] == "warning" for f in result["findings"]) >= 1


# R3 Hugging Face cache snapshots and escapes through the CLI and MCP preflight wrappers.

def snapshot(tmp_path, *, linked_blobs=False):
    """A dataset downloaded without --local-dir: snapshots/<rev>/meta/* link into the same repo cache's blobs/."""
    source = groot_dataset(tmp_path / "plain")
    cache = tmp_path / "cache" / "datasets--org--name"
    blobs = cache / "blobs"
    if linked_blobs:
        (tmp_path / "elsewhere").mkdir()
        cache.mkdir(parents=True)
        blobs.symlink_to(tmp_path / "elsewhere")
    else:
        blobs.mkdir(parents=True)
    revision = cache / "snapshots" / "abc123"
    (revision / "meta").mkdir(parents=True)
    for path in (source / "meta").iterdir():
        blob = blobs / hashlib.sha256(path.read_bytes()).hexdigest()
        blob.write_bytes(path.read_bytes())
        os.symlink(os.path.relpath(blob, revision / "meta"), revision / "meta" / path.name)
    return cache, revision, groot_config(revision)


def test_r3_cli_accepts_a_cache_snapshot_and_refuses_a_symlinked_blobs_directory(tmp_path, capsys):
    _, revision, config = snapshot(tmp_path / "good")
    main(["preflight", str(revision), "--groot-config", str(config), "--json"])
    assert json.loads(capsys.readouterr().out)["components"]["groot_contract"]["status"] == "pass"
    _, revision, config = snapshot(tmp_path / "bad", linked_blobs=True)
    main(["preflight", str(revision), "--groot-config", str(config), "--json"])
    part = json.loads(capsys.readouterr().out)["components"]["groot_contract"]
    assert part["status"] == "unknown" and "escapes the dataset directory" in part["findings"][0]["message"]


def test_r3_mcp_snapshot_authority_and_escapes(tmp_path):
    cache, revision, config = snapshot(tmp_path / "good")
    _, bad_revision, bad_config = snapshot(tmp_path / "bad", linked_blobs=True)
    outside = groot_config(tmp_path, name="outside_config.py")

    def call(roots, source, cfg, response_format="detailed"):
        server = create_server([str(root) for root in roots])
        return structured(asyncio.run(server.call_tool("preflight_check", {
            "source": str(source), "groot_modality_config": str(cfg), "response_format": response_format})))

    # The repository cache directory holds snapshots/ and blobs/: authorized, accepted.
    assert call([cache], revision, config)["components"]["groot_contract"]["status"] == "pass"
    assert call([cache], revision, config, "concise")["components"]["groot_contract"]["status"] == "pass"
    # A snapshot-only root does not reach the blobs: an explicit unknown naming the required root.
    part = call([revision], revision, config)["components"]["groot_contract"]
    message = part["findings"][0]["message"]
    assert part["status"] == "unknown" and "outside the configured read roots" in message
    assert "repository cache directory that holds both snapshots/ and blobs/" in message
    # A symlinked blobs/ leading outside the roots is never opened; inside wider roots the contract refuses it.
    part = call([tmp_path / "bad" / "cache"], bad_revision, bad_config)["components"]["groot_contract"]
    assert part["status"] == "unknown" and "outside the configured read roots" in part["findings"][0]["message"]
    part = call([tmp_path], bad_revision, bad_config)["components"]["groot_contract"]
    assert part["status"] == "unknown" and "escapes the dataset directory" in part["findings"][0]["message"]
    # A config outside the read roots is refused before preflight runs.
    with pytest.raises(Exception, match="outside"):
        call([cache], revision, outside)


def test_r3_actual_stdio_server_snapshot_and_refused_config(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cache, revision, config = snapshot(tmp_path)
    outside = groot_config(tmp_path, name="outside_config.py")

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=["-B", "-m", "policy_doctor", "serve", "--root", str(cache)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src")))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                good = await client.call_tool("preflight_check", {
                    "source": str(revision), "groot_modality_config": str(config), "commanded_reach": True})
                assert not good.isError
                body = good.structuredContent
                assert body["components"]["groot_contract"]["status"] == "pass"
                assert body["components"]["commanded_reach"]["limits"] == engine.RESIDUAL_LIMITS
                assert len(json.dumps(body, allow_nan=False).encode()) <= SUMMARY_LIMIT
                refused = await client.call_tool("preflight_check", {
                    "source": str(revision), "groot_modality_config": str(outside)})
                assert refused.isError and "outside" in refused.content[0].text
    asyncio.run(asyncio.wait_for(run(), timeout=60))


# R4 core install without the dataset extra (no pyarrow).

def test_r4_without_the_dataset_extra_residual_triage_is_unknown_and_contract_error_survives(
        tmp_path, reach, monkeypatch, capsys):
    root, config = combined(tmp_path, reach, modality=SEVEN_DOF), groot_config(tmp_path)
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(ImportError):
        import pyarrow.parquet  # noqa: F401
    result = engine.preflight(str(root), groot_modality_config=str(config), commanded_reach=True)
    part = result["components"]["commanded_reach"]
    assert part["status"] == "unknown" and "policy-deployment-doctor[dataset]" in part["findings"][0]["message"]
    assert part["residual_statement"] == engine.RESIDUAL_STATEMENTS["unknown"]
    # The contract reads JSON metadata only, so its known error stays whole and dominates.
    assert result["components"]["groot_contract"]["status"] == result["status"] == "error"
    assert {"groot.state_slice", "groot.action_slice"} <= {f["code"] for f in result["components"]["groot_contract"]["findings"]}
    assert main(["preflight", str(root), "--groot-config", str(config), "--commanded-reach", "--json"]) == 1
    cli = json.loads(capsys.readouterr().out)
    assert "policy-deployment-doctor[dataset]" in cli["components"]["commanded_reach"]["findings"][0]["message"]


# R5 concise MCP summaries at 8192 bytes keep witnesses and the residual limits.

def test_r5_trimming_shortens_text_before_dropping_witnesses():
    long_path = "/数据/机器人🙂" * 300
    pointer = "/字段" * 200
    findings = ([{"code": f"groot.error{i}", "status": "error", "message": "错误🙂" * 300, "remedy": "修复" * 300,
                  "evidence": [{"path": long_path + str(i), "pointer": pointer, "sha256": f"{i:064x}",
                                "value": "值" * 500}]} for i in range(5)]
                + [{"code": f"reach.w{i}", "status": "warning", "message": "警告" * 300,
                    "evidence": [{"path": "/short/" + str(i), "pointer": "/p", "sha256": "b" * 64}]} for i in range(20)])
    components = {"groot_contract": {"status": "error", "title": "GR00T dataset contract", "consumer": groot_contract.GROOT,
                                     "contract_statement": engine.GROOT_WITHOUT_CONFIG + " " + engine.GROOT_NOT_READ,
                                     "counts": {"error": 5}, "coverage": {"modality_config_supplied": False}},
                  "commanded_reach": {"status": "warning", "title": "Recorded-position residual triage",
                                      "residual_statement": engine.RESIDUAL_STATEMENTS["warning"],
                                      "residual_limits": engine.RESIDUAL_LIMITS, "counts": {"warning": 20},
                                      "coverage": {"rows": 200000, "episodes": 50}}}
    report = {"kind": "dataset_preflight", "status": "error", "scope": "Scope " * 400, "components": components,
              "findings": findings, "counts": {"error": 5, "warning": 20}, "total_findings": 25,
              "next_actions": ["下一步" * 200] * 6}
    summary = _summary(report)
    assert len(json.dumps(summary, allow_nan=False).encode()) <= SUMMARY_LIMIT
    assert summary["counts"] == report["counts"] and summary["status"] == "error"
    assert summary["components"]["commanded_reach"]["limits"] == engine.RESIDUAL_LIMITS
    assert "not a full GR00T compatibility check" in summary["components"]["groot_contract"]["statement"]
    shown = summary["findings_shown"]
    assert len(shown) >= 3 and summary["findings_omitted"] == 25 - len(shown)
    assert summary["next_actions_omitted"] > 0  # generic actions shrink before the three most severe witnesses
    assert [f["status"] for f in shown] == sorted((f["status"] for f in shown), key=["error", "warning"].index)
    for item in shown:
        original = next(f for f in findings if f["code"] == item["code"])["evidence"][0]
        witness = item["evidence"][0]
        assert witness["sha256"] == original["sha256"]
        for key in ("path", "pointer"):
            assert original[key].startswith(witness[key])
            if witness[key] != original[key]:
                assert witness["full_" + key + "_sha256"] == hashlib.sha256(original[key].encode()).hexdigest()
    # A finding is dropped only once every shown finding has been shortened.
    if len(shown) < 12:
        assert all("text_truncated" in item for item in shown)


def test_r5_real_consumer_and_new_components_keep_witnesses_under_unicode_pressure(request):
    job = request.getfixturevalue("consumer_job")
    result = engine.preflight(**consumer_fixtures.request(job, commanded_reach=True, consumer_require_equal=True,
                                                        groot_version="n1.7"))
    result["findings"] += [{"status": "warning", "code": "generated.unicode." + str(i), "message": "机器人🙂" * 400,
                            "evidence": [{"path": "/" + "路径" * 400, "pointer": "/x", "sha256": "c" * 64}]}
                           for i in range(40)]
    concise = _summary(result)
    assert len(json.dumps(concise, allow_nan=False).encode()) <= SUMMARY_LIMIT
    assert concise["status"] == "error" and concise["counts"] == result["counts"]
    compact = concise["components"]["consumer_comparison"]
    assert compact["first_error"]["code"] == "consumer.selected_value_invariant_changed"
    consumer_fixtures.assert_source_witness(compact, consumer_fixtures.parquet_identity(job))
    assert concise["components"]["commanded_reach"]["limits"] == engine.RESIDUAL_LIMITS
    assert concise["components"]["groot_contract"]["statement"]
    assert concise["findings_shown"] and concise["findings_shown"][0]["status"] == "error"
    assert all(item["evidence"] for item in concise["findings_shown"] if item["code"].startswith("generated."))


# R6 a selected contract without a training config never implies full GR00T compatibility.

def test_r6_contract_without_config_states_what_was_not_checked(tmp_path, capsys):
    root = groot_dataset(tmp_path / "ds")
    result = engine.preflight(str(root), groot_version="n1.7")
    part = result["components"]["groot_contract"]
    assert part["status"] != "pass"  # never a pass without the config
    assert part["contract_statement"] == engine.GROOT_WITHOUT_CONFIG + " " + engine.GROOT_NOT_READ
    assert "not a full GR00T compatibility check" in part["contract_statement"]
    assert part["versions_run"] == ["n1.7"]
    assert {"config keys", "camera pairing", "language", "relative actions"} <= set(part["versions"]["n1.7"]["not_checked"])
    main(["preflight", str(root), "--groot-version", "n1.7"])
    assert "    " + engine.GROOT_WITHOUT_CONFIG in capsys.readouterr().out
    statement = _summary(result)["components"]["groot_contract"]["statement"]
    assert "config keys, camera pairing, language and relative actions were not checked" in statement
    supplied = engine.preflight(str(root), groot_version="n1.7",
                                groot_modality_config=str(groot_config(tmp_path)))["components"]["groot_contract"]
    assert supplied["status"] == "pass" and "not a full" not in supplied["contract_statement"]
    assert supplied["contract_statement"] == engine.GROOT_WITH_CONFIG + " " + engine.GROOT_NOT_READ
    assert "config keys" not in supplied["versions"]["n1.7"]["not_checked"]
    assert "training" in supplied["versions"]["n1.7"]["not_checked"]
    broken = groot_config(tmp_path, CONFIG + "\ndef (:\n", "broken.py")
    failed = engine.preflight(str(root), groot_version="n1.7", groot_modality_config=str(broken))["components"]["groot_contract"]
    assert failed["status"] == "unknown" and "did not complete" in failed["contract_statement"]
    with pytest.raises(InputError, match="not a file"):  # a mistyped config path is refused, never dropped
        engine.preflight(str(root), groot_modality_config=str(tmp_path / "absent.py"))


@pytest.mark.parametrize("kwargs", [{"commanded_reach": "true"}, {"groot_version": "auto"},
                                    {"groot_modality_config": "hf://org/config"},
                                    {"groot_modality_config": "/outside/config.py", "read_roots": ["/authorized"]}])
def test_invalid_new_selectors_refuse_before_any_hub_request(monkeypatch, kwargs):
    monkeypatch.setattr(engine.hub, "resolve_dataset_content", forbidden)
    monkeypatch.setattr(engine.hub, "resolve_hub_snapshot", forbidden)
    with pytest.raises(InputError):
        engine.preflight("hf://org/data", fetch=True, **kwargs)


def test_r2_matching_hashes_are_listed_once_and_a_differing_hash_still_reaches_input_stability(tmp_path, monkeypatch):
    root = plain_dataset(tmp_path)
    info = json.loads((root / "meta/info.json").read_text())
    for key in ("action", "observation.state"):
        info["features"][key]["names"] = ["a.pos", "b.pos"]  # motor positions: the residual reader reads the rows
    (root / "meta/info.json").write_text(json.dumps(info))
    result = engine.preflight(str(root), commanded_reach=True)
    part = result["components"]["commanded_reach"]
    parquet = str((root / "data/chunk-000/file-000.parquet").resolve())
    assert part["coverage"]["data_files_read"] == 1 and "input_stability" not in result["components"]
    assert parquet in {a["path"] for a in result["components"]["numeric_rows"]["artifacts"]}
    assert parquet not in {a["path"] for a in part["artifacts"]}
    assert part["coverage"]["files_with_matching_hash_in_other_components"] == 2  # meta/info.json and the rows
    original = reach_module.check_commanded_reach

    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        report["coverage"]["data_files"][0]["sha256"] = "0" * 64
        return report

    monkeypatch.setattr(reach_module, "check_commanded_reach", changed)
    unstable = engine.preflight(str(root), commanded_reach=True)
    assert parquet in {a["path"] for a in unstable["components"]["commanded_reach"]["artifacts"]}
    assert unstable["components"]["input_stability"]["status"] == unstable["status"] == "unknown"
    assert parquet in unstable["components"]["input_stability"]["findings"][0]["message"]


# Actual entry points: `python -m policy_doctor` for every new CLI path, and the real stdio MCP server.

def cli(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", "-m", "policy_doctor", *map(str, args)], capture_output=True,
                          text=True, env=env, cwd=cwd, timeout=300)


def test_entry_point_help_documents_the_new_flags():
    out = cli("preflight", "--help")
    assert out.returncode == 0, out.stderr
    text = " ".join(out.stdout.split()).replace("- ", "-")  # argparse may wrap after a hyphen
    assert "--groot-config PATH" in text and "--groot-config-name NAME" in text and "--commanded-reach" in text
    assert "GR00T training modality config .py, parsed without executing it." in text
    assert "leaving config checks unknown" in text and "known built-in N1.5 data config name" in text
    assert "--groot-version {n1.5,n1.7}" in text and "check both versions without a config" in text
    assert "Opt in to recorded-position residual triage" in text
    assert "Not a calibration, actuator or robot-health check" in text


def test_entry_point_exit_codes_for_every_new_path(tmp_path, reach):
    plain = plain_dataset(tmp_path)
    groot_layout = plain_dataset(tmp_path / "groot_layout")  # passes every existing check, plus meta/modality.json
    (groot_layout / "meta" / "modality.json").write_text(json.dumps(groot_fixtures.MODALITY))
    config = groot_config(tmp_path)
    rows = lerobot_rows(tmp_path, reach, "rows")  # every check passes, residual triage included
    clipped = lerobot_rows(tmp_path, reach, "clipped", reach=reach._clip(4, high=150.0))

    def run(*args, code, cwd=None):
        out = cli("preflight", *args, cwd=cwd)
        assert out.returncode == code, (args, out.returncode, out.stdout[-2000:], out.stderr[-2000:])
        return out

    report = json.loads(run(plain, "--json", code=0).stdout)
    assert not {"groot_contract", "commanded_reach"} & report["components"].keys()
    report = json.loads(run(plain, "--commanded-reach", "--json", code=3).stdout)
    assert report["components"]["commanded_reach"]["findings"][0]["code"] == "reach.not_applicable"
    run(plain, "--commanded-reach", "--fail-on", "warning", code=0)
    run(plain, "--commanded-reach", "--fail-on", "unknown", code=1)
    text = run(groot_layout, "--groot-version", "n1.7", code=1).stdout  # explicit legacy N1.7 path
    assert "GR00T dataset contract: error" in text and engine.GROOT_WITHOUT_CONFIG in text
    report = json.loads(run(groot_layout, "--groot-config", config, "--groot-config-name", "so100_config", "--json",
                            code=1).stdout)
    assert report["components"]["groot_contract"]["modality_config_name"] == "so100_config"
    report = json.loads(run(groot_layout, "--groot-config", config.name, "--json", code=1, cwd=tmp_path).stdout)
    assert report["components"]["groot_contract"]["input_sources"][1] == str(config.resolve())
    assert "not a file" in run(plain, "--groot-config", tmp_path / "absent.py", code=2).stderr
    broken = groot_config(tmp_path, CONFIG + "\ndef (:\n", "broken.py")
    report = json.loads(run(groot_layout, "--groot-version", "n1.7", "--groot-config", broken, "--json", code=3).stdout)
    assert report["components"]["groot_contract"]["status"] == report["status"] == "unknown"
    assert "Cannot parse" in report["components"]["groot_contract"]["findings"][0]["message"]
    report = json.loads(run(rows, "--commanded-reach", "--json", code=0).stdout)
    assert report["components"]["commanded_reach"]["status"] == "pass" and report["status"] == "pass"
    run(rows, "--commanded-reach", "--fail-on", "warning", code=0)
    text = run(clipped, "--commanded-reach", code=0).stdout
    assert "Recorded-position residual triage: warning" in text and engine.RESIDUAL_LIMITS in text
    run(clipped, "--commanded-reach", "--fail-on", "warning", code=1)
    markdown = run(clipped, "--commanded-reach", "--markdown", code=0).stdout
    assert markdown.startswith("# RoboClue dataset preflight") and "Recorded-position residual triage: warning" in markdown
    output = tmp_path / "saved" / "report.json"
    run(clipped, "--commanded-reach", "--json", "--output", output, code=0)
    saved = json.loads(output.read_text())
    assert saved["components"]["commanded_reach"]["status"] == saved["status"] == "warning"
    for args in ([plain, "--groot-version", "n1.7", "--groot-config-name", "so100"], [plain, "--groot-config", "hf://org/config"],
                 [plain, "--groot-config", ""], ["hf://org/data", "--commanded-reach"],
                 [plain, "--commanded-reach=yes"], [plain, "--groot-config"]):
        out = run(*args, code=2)
        assert not out.stdout
    assert "requires --groot-config" in run(plain, "--groot-version", "n1.7", "--groot-config-name", "so100", code=2).stderr


def test_actual_stdio_server_schema_and_new_parameters(tmp_path, reach):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root, config, plain = combined(tmp_path, reach), groot_config(tmp_path), plain_dataset(tmp_path)

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=["-B", "-m", "policy_doctor", "serve", "--root", str(tmp_path)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src")))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                assert len(tools) == 29
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in tools}
                schema = next(t for t in tools if t.name == "preflight_check").inputSchema
                assert schema["additionalProperties"] is False
                properties, required = schema["properties"], set(schema.get("required", []))
                assert properties["commanded_reach"]["type"] == "boolean"
                assert "robot health" in next(t for t in tools if t.name == "preflight_check").description
                for name in ("groot_modality_config", "groot_config_name"):
                    assert {"type": "string"} in properties[name]["anyOf"]
                    assert {"type": "null"} in properties[name]["anyOf"]
                assert {"type": "string", "enum": ["n1.5", "n1.7"]} in properties["groot_version"]["anyOf"]
                assert {"type": "null"} in properties["groot_version"]["anyOf"]
                assert properties["groot_version"]["default"] is None
                assert not {"commanded_reach", "groot_modality_config", "groot_config_name", "groot_version"} & required
                plain_body = (await client.call_tool("preflight_check", {"source": str(plain)})).structuredContent
                assert not {"groot_contract", "commanded_reach"} & plain_body["components"].keys()
                arguments = {"source": str(root), "groot_version": "n1.7", "groot_modality_config": str(config),
                             "groot_config_name": "so100_config", "commanded_reach": True}
                short = await client.call_tool("preflight_check", arguments)
                assert not short.isError
                body = short.structuredContent
                assert len(json.dumps(body, allow_nan=False).encode()) <= SUMMARY_LIMIT
                assert body["components"]["groot_contract"]["status"] == "pass"
                assert body["components"]["groot_contract"]["statement"].startswith(engine.GROOT_WITH_CONFIG)
                assert body["components"]["commanded_reach"]["residual_statement"] == PASS_STATEMENT
                assert body["components"]["commanded_reach"]["limits"] == engine.RESIDUAL_LIMITS
                full = await client.call_tool("preflight_check", {**arguments, "response_format": "detailed"})
                assert not full.isError
                detailed = full.structuredContent
                assert detailed["status"] == body["status"] and detailed["counts"] == body["counts"]
                raw = engine.preflight(str(root), groot_version="n1.7", groot_modality_config=str(config),
                    groot_config_name="so100_config", commanded_reach=True, read_roots=[str(tmp_path)])
                assert raw["components"]["commanded_reach"]["joints"]
                if "truncated_from_bytes" in detailed:
                    # Default detailed responses may exceed192KiB on long checkout paths.
                    assert detailed["truncated_from_bytes"] > server_module.RESULT_LIMIT
                    assert detailed["components"]["commanded_reach"]["status"] == \
                        raw["components"]["commanded_reach"]["status"]
                    assert detailed["components"]["commanded_reach"]["residual_statement"] == PASS_STATEMENT
                    assert detailed["full_report_sha256"] == hashlib.sha256(json.dumps(
                        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                        allow_nan=False).encode()).hexdigest()
                else:
                    assert len(json.dumps(raw).encode()) <= server_module.RESULT_LIMIT
                    assert detailed["components"]["commanded_reach"]["joints"]
                    assert detailed["components"]["groot_contract"]["input_sources"] == [str(root.resolve()),
                                                                                         str(config.resolve())]
                for changes, expected in (({"commanded_reach": "yes"}, "strict boolean"),
                                          ({"commanded_reach": 1}, "strict boolean"),
                                          ({"groot_modality_config": None, "groot_config_name": "so100_config"},
                                           "requires --groot-config"),
                                          ({"groot_modality_config": "/etc/hostname"}, "outside"),
                                          ({"groot_modality_config": "hf://org/config"}, "local"),
                                          ({"groot_config": str(config)}, "groot_config")):
                    rejected = await client.call_tool("preflight_check", {**arguments, **changes})
                    assert rejected.isError and expected in rejected.content[0].text, (changes, rejected.content)
    asyncio.run(asyncio.wait_for(run(), timeout=120))
