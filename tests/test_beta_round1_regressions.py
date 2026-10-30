"""Round-1 simulated-beta bugs still present in 0.38 (`.validation/beta-p2-038`), each with the tester's repro."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor import hub
from policy_doctor.artifacts import InputError, load_checkpoint, load_dataset
from policy_doctor.cli import main, parser
from policy_doctor.investigation import investigate_run


def _dataset(root):
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 10, "robot_type": "so101_follower", "features": {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]}}}))
    pq.write_table(pa.table({"task_index": [0], "task": ["pick"]}), root / "meta" / "tasks.parquet")
    return root


def _checkpoint(root):
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"type": "act", "input_features": {
        "observation.state": {"type": "STATE", "shape": [2]}}, "output_features": {"action": {"type": "ACTION", "shape": [2]}}}))
    return root


def _hub(monkeypatch, public, missing_revisions=(), gated=None):
    gated = gated or {}
    """Anonymous Hub: 401 for any repository that is not public under the requested kind, as the real API answers,
    and 404 for a public repository's missing revision."""
    def request(url, *, limit, required_sha=None):
        kind, path = url.split("/api/", 1)[1].split("/", 1)
        repo = "/".join(path.split("/")[:2])
        revision = path.split("/revision/", 1)[1] if "/revision/" in path else None
        code = 401 if (kind, repo) not in public else 404 if revision in missing_revisions else None
        if code:
            error = hub.HubUnavailable(f"Public Hub metadata unavailable (HTTP {code}); no credentials used")
            error.http_status = code
            raise error
        return json.dumps({"id": repo, "private": False, "gated": gated.get((kind, repo), False)}).encode()
    monkeypatch.setattr(hub, "_request", request)


def test_dataset_given_as_check_source_is_named_not_reported_as_401(monkeypatch):
    # Chen and Maya: `check hf://lerobot/berkeley_autolab_ur5 --fetch` said "HTTP 401" for a public dataset.
    _hub(monkeypatch, {("datasets", "org/data")})
    finding = hub.check_checkpoint("hf://org/data", fetch=True)["findings"][0]
    assert finding["code"] == "hub.metadata_unavailable"
    assert "A public dataset repository exists at this ID" in finding["message"]
    assert "preflight hf://org/data --fetch" in finding["message"]


def test_shared_id_with_a_missing_model_revision_is_not_called_a_dataset(monkeypatch):
    # Codex review of 84033c8: a model and a dataset can share an ID; a missing model revision is not proof that the
    # ID names only a dataset.
    _hub(monkeypatch, {("models", "org/both"), ("datasets", "org/both")}, missing_revisions={"v9"})
    message = hub.check_checkpoint("hf://org/both", fetch=True, revision="v9")["findings"][0]["message"]
    assert "HTTP 404" in message and "lacks that revision" in message
    assert "not a policy checkpoint" not in message and "if you meant it" in message


def test_missing_repository_names_the_possible_causes(monkeypatch):
    # Sam: a nonexistent repository gave only "HTTP 401; no credentials used".
    _hub(monkeypatch, set())
    message = hub.check_checkpoint("hf://org/typo", fetch=True)["findings"][0]["message"]
    assert "HTTP 401" in message and "does not exist, is private or gated" in message and "Check the model ID" in message


def test_model_given_as_the_dataset_is_named(monkeypatch):
    _hub(monkeypatch, {("models", "org/model")})
    with pytest.raises(hub.HubUnavailable, match="A public model repository exists at this ID"):
        with hub.resolve_hub_snapshot("hf://org/model", repo_type="dataset", fetch=True):
            pass


def test_dataset_folder_given_as_a_checkpoint_says_so(tmp_path):
    # Arjun, Lena, Maya: "Expected a JSON file no larger than 8388608 bytes: .../config.json".
    ds = _dataset(tmp_path / "pusht")
    with pytest.raises(InputError, match="is a LeRobot dataset .* not a checkpoint.*preflight"):
        load_checkpoint(ds)
    with pytest.raises(InputError, match="is a policy checkpoint .* not a dataset"):
        load_dataset(_checkpoint(tmp_path / "ckpt"))
    with pytest.raises(InputError, match="No config.json in .*pretrained_model"):
        load_checkpoint(tmp_path)


@pytest.mark.parametrize("command", [["inspect", "hf://Org/repo"], ["audit", "hf://Org/repo"]])
def test_hub_uri_to_a_local_command_is_explained(command, tmp_path, monkeypatch, capsys):
    # Arjun and Maya: the URI became ".../hf:/Org/repo/config.json".
    monkeypatch.chdir(tmp_path)
    assert main(command) == 2
    message = json.loads(capsys.readouterr().err)["error"]["message"]
    assert message.startswith("hf:// sources are read only with --fetch") and "hf:/" not in message.split("--fetch", 1)[1]


def test_missing_case_directory_is_named(tmp_path, capsys):
    # Sam: reproduce_incident_case on a typo said "Expected a JSON file ... manifest.json".
    assert main(["reproduce-case", str(tmp_path / "shared-chek")]) == 2
    message = json.loads(capsys.readouterr().err)["error"]["message"]
    assert message.startswith("Directory not found: ") and "shared-chek" in message


def test_investigate_does_not_call_dataset_metadata_a_recording(tmp_path):
    # Lena, Maya, Sam: meta/tasks.parquet was listed as a "recording" and the dataset checks were never suggested.
    ds = _dataset(tmp_path / "svla_bad_span")
    result = investigate_run(str(ds))
    assert result["inventory"]["datasets"] == [str(ds)] and result["inventory"]["recordings"] == []
    assert result["next_actions"][0] == {"tool": "preflight_check", "arguments": {"source": str(ds)},
                                         "purpose": result["next_actions"][0]["purpose"]}
    assert "run policy-doctor preflight on each dataset" in result["summary"]


def test_a_json_list_in_a_run_folder_does_not_block_the_audit(tmp_path):
    # Arjun: a `[]` file (or the Hub's episodes_tree.json listing) beside the dataset made investigate run 0 audits.
    run = tmp_path / "run"
    _dataset(run / "dataset")
    _checkpoint(run / "checkpoint")
    (run / "dataset" / "episodes_tree.json").write_text("[]")
    result = investigate_run(str(run))
    assert len(result["audits"]) == 1 and result["intake_problems"] == []
    (run / "dataset" / "broken.json").write_text("{")
    assert investigate_run(str(run))["intake_problems"]  # malformed JSON is still reported


def test_markdown_paths_copy_exactly(tmp_path, capsys):
    # Sam: Markdown printed svla\_bad\_span, so copied paths broke.
    ds = _dataset(tmp_path / "svla_bad_span")
    main(["investigate", str(ds), "--markdown"])
    out = capsys.readouterr().out
    assert f"`{ds}`" in out and "svla\\_bad\\_span" not in out


def test_template_takes_the_training_dataset(tmp_path, capsys):
    ds, cp = _dataset(tmp_path / "ds"), _checkpoint(tmp_path / "cp")
    assert main(["template", str(cp), "--dataset", str(ds)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["features"]["observation.state"]["names"] == ["a", "b"]
    assert value["fps"] == 10 and value["robot_type"] == "so101_follower"


def test_scan_reads_a_typical_dataset_completely_by_default():
    # Lena: the default scan stopped at 10,000 of pusht's 25,650 rows.
    assert parser().parse_args(["scan", "x"]).max_rows == 200000


def test_dataset_hint_survives_the_terminal_view(monkeypatch, capsys):
    # Validator 2a: the 420-character clip dropped the hint from the testers' exact B1 command.
    from policy_doctor.cli import main as cli_main
    _hub(monkeypatch, {("datasets", "lerobot/berkeley_autolab_ur5")})
    cli_main(["check", "hf://lerobot/berkeley_autolab_ur5", "--fetch", "--scope", "metadata"])
    assert "policy-doctor preflight hf://lerobot/berkeley_autolab_ur5 --fetch" in capsys.readouterr().out


def test_gated_repository_of_the_other_kind_is_not_called_public(monkeypatch):
    # Validator 2b.
    _hub(monkeypatch, {("datasets", "org/g")}, gated={("datasets", "org/g"): "auto"})
    message = hub.check_checkpoint("hf://org/g", fetch=True)["findings"][0]["message"]
    assert "A gated or private dataset repository exists" in message and "public dataset" not in message
    assert "preflight" not in message  # no command is suggested for a repository RoboClue refuses


def test_corrupt_recognized_candidates_still_block_a_clean_verdict(tmp_path):
    # Validator 4 (P4): a null config.json or a list meta/info.json is a corrupt candidate, not an unrelated file.
    run = tmp_path / "run"
    _dataset(run / "dataset")
    _checkpoint(run / "checkpoint")
    (run / "cp_new").mkdir()
    (run / "cp_new" / "config.json").write_text("null")
    result = investigate_run(str(run))
    assert result["status"] != "pass" and any("cp_new" in p["path"] for p in result["intake_problems"])
    (run / "cp_new" / "config.json").unlink()
    (run / "dataset" / "meta" / "info.json").write_text("[]")
    result = investigate_run(str(run))
    assert result["status"] != "pass" and any("info.json" in p["path"] for p in result["intake_problems"])


def test_unrelated_list_json_is_listed_in_coverage(tmp_path):
    run = tmp_path / "run"
    _dataset(run / "dataset")
    _checkpoint(run / "checkpoint")
    (run / "dataset" / "episodes_tree.json").write_text("[]")
    result = investigate_run(str(run))
    assert result["coverage"]["non_object_json_skipped"] == [str(run / "dataset" / "episodes_tree.json")]


def test_clip_keeps_text_when_there_is_no_boundary():
    # Validator 6a: "a. " + "b" * 600 used to collapse to "a. ...".
    from policy_doctor.check_report import _clip
    clipped = _clip("a. " + "b" * 600, 420)
    assert len(clipped) <= 420 and clipped.endswith("b ...") and clipped.count("b") > 300


def test_null_file_with_an_arbitrary_name_stays_an_intake_problem(tmp_path):
    # Validator recheck R1: a corrupt deployment named by the user (lekiwi_good.json = null) must not vanish.
    run = tmp_path / "run"
    _dataset(run / "dataset")
    _checkpoint(run / "checkpoint")
    (run / "lekiwi_good.json").write_text("null")
    result = investigate_run(str(run))
    assert result["status"] != "pass" and any("lekiwi_good.json" in p["path"] for p in result["intake_problems"])


def test_list_or_null_json_is_decided_from_one_bounded_read(tmp_path, monkeypatch):
    # Codex review of the R1 follow-up: no second, unbounded read to tell a list from null.
    import pathlib
    run = tmp_path / "run"
    _dataset(run / "dataset")
    _checkpoint(run / "checkpoint")
    (run / "dataset" / "episodes_tree.json").write_text("[]")
    (run / "lekiwi_good.json").write_text("null")
    reads = {}
    original_text, original_bytes = pathlib.Path.read_text, pathlib.Path.read_bytes

    def counted(method):
        def wrapper(self, *args, **kwargs):
            reads[self.name] = reads.get(self.name, 0) + 1
            return method(self, *args, **kwargs)
        return wrapper
    monkeypatch.setattr(pathlib.Path, "read_text", counted(original_text))
    monkeypatch.setattr(pathlib.Path, "read_bytes", counted(original_bytes))
    result = investigate_run(str(run))
    assert reads["episodes_tree.json"] == 1 and reads["lekiwi_good.json"] == 1
    assert result["coverage"]["non_object_json_skipped"] == [str(run / "dataset" / "episodes_tree.json")]
    assert any("lekiwi_good.json" in p["path"] for p in result["intake_problems"])
