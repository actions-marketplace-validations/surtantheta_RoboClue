"""The existing investigate entry point on generated native-boundary evidence."""

import pytest
from test_named_actions import change, contract
from test_named_actions import packet as packet_fixture
from test_named_recording import recording as recording_fixture

from policy_doctor.investigation import investigate_run


@pytest.fixture(name="packet")
def named_packet(tmp_path, monkeypatch):
    return packet_fixture.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture(name="recording")
def native_recording(tmp_path, monkeypatch):
    return recording_fixture.__wrapped__(tmp_path, monkeypatch)


def test_natural_named_capture_frontdoor_uses_native_audit_without_findings_assumption(packet):
    root = packet()
    result = investigate_run(str(root))
    assert result["status"] == "unknown"
    assert result["audits"][0]["kind"] == "named_action_audit"
    assert result["audits"][0]["driver_change_events"] == 1
    assert result["audits"][0]["physical_execution_status"] == "unknown"
    assert any(action["tool"] == "audit_runtime_capture" for action in result["next_actions"])


def test_existing_recorded_capture_does_not_infer_nested_dataset_as_processor_input(recording):
    root = recording()
    result = investigate_run(
        str(root), named_contract={**contract(), "require_recorded_matches_driver_return": True}
    )
    assert result["status"] == "error"
    audit = result["audits"][0]
    assert audit["recording"]["persistence_status"] == "pass"
    assert audit["acceptance_contract"]["violations"][0]["field"] == "recorded_matches_driver_return"
    action = next(action for action in result["next_actions"] if action["tool"] == "audit_runtime_capture")
    assert "dataset" not in action["arguments"]
    assert action["arguments"]["named_contract"]["require_recorded_matches_driver_return"] is True


def test_reference_first_supported_named_role_difference_is_observation(packet):
    reference, candidate = packet("reference", limited=False), packet("candidate")
    result = investigate_run(str(candidate), reference=str(reference))
    comparison = result["comparison"]
    assert comparison["comparison_supported"] is True
    assert comparison["difference_is_fault"] is False
    assert comparison["first_supported_difference"]["stage"] == "driver_return"
    assert comparison["first_supported_difference"]["tick_ordinal"] == 0
    assert comparison["first_supported_difference"]["candidate"]["a.pos"] == 1.0
    assert "ordinal" in comparison["alignment"] and "physical" in comparison["alignment"]
    assert all(
        e["sha256"] and e["pointer"].startswith("/ticks/0/")
        for e in comparison["first_supported_difference"]["evidence"]
    )
    assert result["status"] == "unknown"  # No consumer requirement was supplied.


def test_effective_robot_processor_difference_precedes_driver_difference(packet):
    reference, candidate = packet("reference"), packet("candidate")

    def mutate(data):
        for role in (
            "robot_processor_output",
            "wrapper_request",
            "driver_request",
            "driver_return",
            "wrapper_return",
        ):
            data["ticks"][0][role]["a.pos"] += 1.0

    change(candidate, mutate)
    comparison = investigate_run(str(candidate), reference=str(reference))["comparison"]
    assert comparison["first_supported_difference"]["stage"] == "robot_processor_output"
    assert comparison["difference_is_fault"] is False


def test_recorded_correction_first_return_delta_and_later_row_keeps_native_evidence(recording):
    before = recording("before")
    corrected = recording("corrected", recorded_corrected=True)
    result = investigate_run(
        str(before), reference=str(corrected), named_contract={"require_recorded_matches_driver_return": True}
    )
    assert result["status"] == "error"
    comparison = result["comparison"]
    assert comparison["first_supported_difference"]["stage"] == "canonical_return"
    assert comparison["recording_comparison"]["comparison_supported"] is True
    assert comparison["recording_comparison"]["equal"] is False
    assert any(
        e.get("column") == "action"
        for e in comparison["recording_comparison"]["first_difference"]["evidence"]
    )


def test_ambiguous_reference_never_auto_pairs(packet, tmp_path):
    candidate = packet("candidate")
    reference = tmp_path / "references"
    reference.mkdir()
    packet("references/first")
    packet("references/second")
    result = investigate_run(str(candidate), reference=str(reference))
    assert result["comparison"]["comparison_supported"] is False
    assert result["comparison"]["status"] == "unknown"
    assert result["comparison"]["first_supported_difference"] is None


def test_missing_role_preserves_unknown_instead_of_inventing_first_stage(packet):
    reference, candidate = packet("reference"), packet("candidate", offset=1.0)
    change(candidate, lambda d: d["ticks"][0].update(robot_processor_output=None))
    result = investigate_run(str(candidate), reference=str(reference))
    assert result["comparison"]["comparison_supported"] is False
    assert result["comparison"]["status"] == "unknown"


def test_named_criterion_strict_guard_is_not_silently_unused(packet):
    from policy_doctor.artifacts import InputError

    with pytest.raises(InputError, match="Boolean"):
        investigate_run(str(packet()), named_contract={"require_recorded_matches_driver_return": 1})


def test_input_native_bytes_unchanged_after_reference_investigation(packet):
    reference, candidate = packet("reference"), packet("candidate")
    before = {
        str(path): path.read_bytes()
        for root in (reference, candidate)
        for path in root.rglob("*")
        if path.is_file()
    }
    investigate_run(str(candidate), reference=str(reference), named_contract=contract())
    assert all(__import__("pathlib").Path(path).read_bytes() == value for path, value in before.items())


def test_layout_family_frontdoor_preserves_raw_scope_without_processor_findings(tmp_path):
    from test_layouts import capture

    result = investigate_run(str(capture(tmp_path / "layout")))
    assert result["status"] in ("pass", "unknown")
    assert result["audits"][0]["kind"] == "resolved_layout_audit"
    assert result["audits"][0]["raw_binding"]["status"] == "pass"


def test_command_family_frontdoor_preserves_unknown_consumer_criterion(tmp_path, monkeypatch):
    from test_command_path import packet as command_fixture

    root = command_fixture.__wrapped__(tmp_path, monkeypatch)()
    result = investigate_run(str(root))
    assert result["status"] == "unknown"
    assert result["audits"][0]["kind"] == "command_path_audit"
    assert result["audits"][0]["numeric_replay_status"] == "pass"


def test_cli_one_entry_reference_criterion_and_compact_reports(recording, tmp_path, capsys):
    import json

    from policy_doctor.cli import main

    before, after = recording("before"), recording("after", recorded_corrected=True)
    criterion = tmp_path / "criterion.json"
    criterion.write_text(json.dumps({"require_recorded_matches_driver_return": True}))
    args = ["investigate", str(before), "--reference", str(after), "--named-contract", str(criterion)]
    assert main([*args, "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["comparison"]["first_supported_difference"]["stage"] == "canonical_return"
    assert main(args) == 1
    text = capsys.readouterr().out
    assert "First supported difference" in text and "persistence pass" in text
    assert "Physical execution" in text and len(text) < 10000
    assert main([*args, "--markdown"]) == 1
    markdown = capsys.readouterr().out
    assert "Observed native roles" in markdown and "SHA-256" in markdown
    assert "row=0" in markdown or "row" in markdown


def test_mcp_authorizes_reference_before_any_candidate_intake(packet, tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from policy_doctor import server as module
    from policy_doctor.artifacts import InputError

    tools = module.create_server([str(tmp_path)])._tool_manager._tools
    assert len(tools) == 29
    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
    called = []
    original = module.investigate_run

    def observe(*args, **kwargs):
        called.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "investigate_run", observe)
    candidate, reference = packet("candidate"), packet("reference")
    investigate = tools["investigate_run_directory"].fn
    with pytest.raises(InputError, match="outside"):
        investigate(str(candidate), reference="/outside")
    assert called == []
    report = investigate(str(candidate), reference=str(reference), named_contract=contract())
    assert len(called) == 1 and report["comparison"]["comparison_supported"]
    with pytest.raises(InputError, match="Boolean"):
        investigate(str(candidate), named_contract={"require_recorded_matches_driver_return": 1})


def test_existing_compare_endpoint_reuses_same_named_comparison(packet):
    from policy_doctor.runtime_diff import compare_runtime_captures

    reference, candidate = packet("reference", limited=False), packet("candidate")
    result = compare_runtime_captures(str(reference), str(candidate))
    assert result["kind"] == "run_boundary_comparison"
    assert result["first_supported_difference"]["stage"] == "driver_return"
    assert result["difference_is_fault"] is False


def test_same_frontdoor_delegates_existing_isaac_comparison(tmp_path):
    from test_layouts import capture

    first, second = capture(tmp_path / "first"), capture(tmp_path / "second", ids=(1, 0))
    result = investigate_run(str(second), reference=str(first))
    assert result["comparison"]["kind"] == "resolved_layout_comparison"
    assert result["comparison"]["ordered_layout_equal"] is False


def test_existing_case_folder_gets_reproduction_guidance_without_running_code(packet, tmp_path):
    from policy_doctor.incidents import export_incident_case

    before, after = packet("before"), packet("after", limited=False)
    case = tmp_path / "case"
    export_incident_case("native_named_action", before, case, contract(require=True), corrected=after)
    result = investigate_run(str(case))
    assert result["status"] == "unknown"
    assert result["inventory"]["incident_cases"] == [str(case)]
    action = next(action for action in result["next_actions"] if action["tool"] == "reproduce_incident_case")
    assert action["arguments"] == {"directory": str(case)}
    assert not result["audits"]  # Before/after remain distinct candidates, never auto-selected.


def test_compare_cli_human_and_markdown_preserve_unknown_fault_scope(packet, capsys):
    from policy_doctor.cli import main

    reference, candidate = packet("reference", limited=False), packet("candidate")
    args = ["compare-captures", str(reference), str(candidate)]
    assert main(args) == 3
    assert "driver_return" in capsys.readouterr().out
    assert main([*args, "--markdown"]) == 3
    rendered = capsys.readouterr().out
    assert "Observed native boundary comparison" in rendered
    assert "not by themselves establish a fault" in rendered


def test_capped_reference_discovery_never_certifies_pairing(packet):
    reference, candidate = packet("reference"), packet("candidate")
    result = investigate_run(str(candidate), reference=str(reference), max_files=1)
    assert result["comparison"]["comparison_supported"] is False
    assert result["status"] == "unknown"


def test_reference_intake_mutation_refused_before_comparison(packet, monkeypatch):
    from policy_doctor import named_actions

    reference, candidate = packet("reference"), packet("candidate")
    original = named_actions._recheck
    called = 0

    def modified(root, hashes):
        nonlocal called
        called += 1
        if called == 4:
            change(reference, lambda d: d.update(external_change=True))
        return original(root, hashes)

    monkeypatch.setattr(named_actions, "_recheck", modified)
    result = investigate_run(str(candidate), reference=str(reference))
    assert result["status"] == "unknown"
    assert result["comparison"]["comparison_supported"] is False
    assert result["intake_problems"]


def test_native_run_folder_capsule_contents_are_not_extra_dataset_candidates(recording, tmp_path):
    import shutil

    root = tmp_path / "run"
    root.mkdir()
    capture = recording("run/capture")
    shutil.copytree(capture / "dataset", root / "native/dataset")
    result = investigate_run(str(root), named_contract={"require_recorded_matches_driver_return": True})
    assert result["status"] == "error"
    assert result["coverage"]["truncated"] is False
    assert result["inventory"]["captures"] == [str(capture)]
    assert result["inventory"]["datasets"] == [str(root / "native/dataset")]
    assert result["audits"][0]["recording"]["persistence_status"] == "pass"


def test_reference_mixed_checkpoint_and_capture_remains_ambiguous(packet, tmp_path):
    import json

    candidate = packet("candidate")
    parent = tmp_path / "reference"
    parent.mkdir()
    packet("reference/capture")
    (parent / "config.json").write_text(json.dumps({"type": "act", "input_features": {}}))
    result = investigate_run(str(candidate), reference=str(parent))
    assert result["comparison"]["comparison_supported"] is False
    assert result["comparison"]["first_supported_difference"] is None


def test_named_contract_on_checkpoint_returns_scoped_intake_problem(case):
    result = investigate_run(
        str(case["checkpoint"].parent), named_contract={"require_recorded_matches_driver_return": True}
    )
    assert result["status"] == "unknown"
    assert "Named criteria require" in result["intake_problems"][0]["reason"]


def test_changed_preview_includes_ninth_named_channel_instead_of_unchanged_prefix(packet):
    from policy_doctor.runtime_diff import compare_runtime_captures
    reference, candidate = packet('reference'), packet('candidate')
    def expand(data):
        names = [f'joint{index}.pos' for index in range(9)]
        data['action_names'] = data['engine_action_names'] = names
        tick = data['ticks'][0]
        tick['interpolated'] = list(map(float, range(9)))
        for role in ('canonical_at_processor', 'canonical_at_return', 'robot_processor_output',
                     'wrapper_request', 'driver_request', 'driver_return', 'wrapper_return'):
            tick[role] = dict(zip(names, tick['interpolated'], strict=True))
    change(reference, expand)
    change(candidate, expand)
    def ninth(data):
        data['ticks'][0]['interpolated'][8] = 100.
        for role in ('canonical_at_processor', 'canonical_at_return', 'robot_processor_output',
                     'wrapper_request', 'driver_request', 'driver_return', 'wrapper_return'):
            data['ticks'][0][role]['joint8.pos'] = 100.
    change(candidate, ninth)
    difference = compare_runtime_captures(str(reference), str(candidate))['first_supported_difference']
    assert difference['reference'] == {'joint8.pos': 8.}
    assert difference['candidate'] == {'joint8.pos': 100.}
    assert difference['changed_channel_count'] == 1


def test_writer_tick_zero_precedes_later_runtime_difference(recording):
    import copy

    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_named_recording import refresh_dataset_file
    reference, candidate = recording('reference', recorded_corrected=True), recording('candidate', recorded_corrected=True)
    def second_tick(data):
        extra = copy.deepcopy(data['ticks'][0])
        extra['tick_id'] = 1
        extra['emitted_policy_action'] = False
        data['ticks'].append(extra)
    change(reference, second_tick)
    change(candidate, second_tick)
    # Both generated rows are within the individual native-replay tolerance;
    # their pairwise delta exceeds it. No genuine writer execution is claimed.
    for root, offset in ((reference, -0.000009), (candidate, 0.000009)):
        def mutation(data, offset=offset):
            row = data['recording']['rows'][0]
            for role in ('frame_action', 'action', 'buffer_action'):
                row[role][0] += offset
        change(root, mutation)
        def rows(path, offset=offset):
            values = pq.read_table(path).to_pydict()
            values['action'][0][0] += offset
            values['action'] = pa.array(values['action'], type=pa.list_(pa.float32(), 2))
            pq.write_table(pa.table(values), path)
        refresh_dataset_file(root, 'dataset/data/chunk-000/file-000.parquet', rows)
    def later_change(data):
        tick = data['ticks'][1]
        tick['interpolated'][0] += 2.
        for role in ('canonical_at_processor', 'canonical_at_return', 'robot_processor_output',
                     'wrapper_request', 'driver_request', 'driver_return', 'wrapper_return'):
            tick[role]['a.pos'] += 2.
    change(candidate, later_change)
    comparison = investigate_run(str(candidate), reference=str(reference))['comparison']
    assert comparison['first_supported_difference']['tick_ordinal'] == 0
    assert comparison['first_supported_difference']['stage'] == 'native_writer_row'
