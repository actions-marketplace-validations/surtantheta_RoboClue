"""Same researcher requirement through existing investigation/case interfaces."""

import json

import pytest
from layout_test_support import layout_input
from test_named_actions import packet as packet_fixture

from policy_doctor.artifacts import InputError
from policy_doctor.investigation import investigate_run
from policy_doctor.layouts import write_layout_capture


@pytest.fixture
def layouts(tmp_path):
    result = {}
    for key, order, identity in [
        ("before", ("right_foot", "left_foot"), True),
        ("corrected", ("left_foot", "right_foot"), True),
        ("unknown", ("left_foot", "right_foot"), False),
        ("later", ("left_foot", "right_foot"), True),
    ]:
        metadata, arrays, sources = layout_input(order=order, identity=identity)
        if key == "later":
            arrays = {name: value * 2 for name, value in arrays.items()}
        root = tmp_path / key
        write_layout_capture(root, metadata, arrays, sources=sources)
        result[key] = root
    return result


def layout_requirement(**overrides):
    return {
        "kind": "resolved-layout",
        "parameters": {"expected_names": ["left_foot", "right_foot"], **overrides},
    }


def test_one_question_explicit_body_order_changes_investigate_exit_semantics(layouts):
    root, reference = map(str, (layouts["before"], layouts["corrected"]))
    assert investigate_run(root, reference=reference)["status"] == "pass"  # Existing raw replay alone.
    requirements = layout_requirement()
    result = investigate_run(root, reference=reference, requirements=requirements)
    assert result["status"] == "error"
    assert result["applied_requirements"] == requirements
    assert result["audits"][0]["raw_binding"]["status"] == "pass"
    assert result["audits"][0]["acceptance_contract"]["status"] == "error"
    assert result["comparison"]["reference_criterion_status"] == "pass"
    assert result["comparison"]["named_values_equal"] is True
    assert requirements == layout_requirement()  # Caller criteria are never mutated.


def test_same_requirement_healthy_and_missing_namespace_stay_distinct(layouts):
    requirement = layout_requirement(expected_identity={"prim_path": "/World/envs/env_.*/Robot"})
    for key, expected in [("corrected", "pass"), ("unknown", "unknown"), ("later", "pass")]:
        report = investigate_run(
            str(layouts[key]), reference=str(layouts["corrected"]), requirements=requirement
        )
        assert report["status"] == expected


def test_valid_correction_emits_existing_creator_recipe_without_writing(layouts):
    result = investigate_run(
        str(layouts["before"]), reference=str(layouts["corrected"]), requirements=layout_requirement()
    )
    recipe = result["case_recipe"]
    assert recipe["tool"] == "create_incident_case"
    assert recipe["arguments"]["check"] == "resolved_layout"
    assert recipe["arguments"]["source"] == str(layouts["before"])
    assert recipe["arguments"]["corrected"] == str(layouts["corrected"])
    assert recipe["arguments"]["parameters"]["expected_names"] == ["left_foot", "right_foot"]
    assert recipe["required_inputs"] == ["destination"]
    assert "destination" not in recipe["arguments"]
    assert not (layouts["before"].parent / "case").exists()


def test_incompatible_or_unknown_reference_never_becomes_supplied_correction(layouts):
    for reference in ("before", "unknown"):
        report = investigate_run(
            str(layouts["before"]),
            reference=str(layouts[reference]),
            requirements=layout_requirement(expected_identity={"prim_path": "/World/envs/env_.*/Robot"}),
        )
        assert report["case_recipe"] is None


@pytest.mark.parametrize(
    "requirements",
    [
        [],
        {},
        {"kind": "resolved-layout"},
        {"kind": "resolved-layout", "parameters": {}, "extra": 1},
        {"kind": "trajectory-sampling", "parameters": {"sample_times": [0.0]}},
        {"kind": "resolved-layout", "parameters": {"arbitrary": True}},
        {"kind": "resolved-layout", "parameters": {"expected_names": ["left_foot"], "atol": True}},
        {
            "kind": "resolved-layout",
            "parameters": {"expected_names": ["left_foot"], "expected_identity": True},
        },
    ],
)
def test_requirement_schema_and_raw_values_are_not_coerced(layouts, requirements):
    with pytest.raises(InputError):
        investigate_run(str(layouts["before"]), requirements=requirements)


def test_named_requirements_reuse_existing_contract_and_refuse_conflicting_alias(tmp_path, monkeypatch):
    packet = packet_fixture.__wrapped__(tmp_path, monkeypatch)
    before, after = packet("before"), packet("after", limited=False)
    parameters = {"require_canonical_matches_driver_return": True}
    requirements = {"kind": "native-named-action", "parameters": parameters}
    result = investigate_run(str(before), reference=str(after), requirements=requirements)
    assert result["status"] == "error" and result["comparison"]["reference_criterion_status"] == "pass"
    assert result["case_recipe"]["arguments"]["check"] == "native_named_action"
    assert investigate_run(str(before), named_contract=parameters)["status"] == "error"
    with pytest.raises(InputError, match="conflict|Choose|together"):
        investigate_run(str(before), named_contract=parameters, requirements=requirements)
    with pytest.raises(InputError):
        investigate_run(str(before), requirements=layout_requirement())


def test_cli_same_requirements_file_investigate_and_existing_case_creation(layouts, tmp_path, capsys):
    from policy_doctor.cli import main

    path = tmp_path / "requirements.json"
    path.write_text(json.dumps(layout_requirement()))
    argv = [
        "investigate",
        str(layouts["before"]),
        "--reference",
        str(layouts["corrected"]),
        "--requirements",
        str(path),
    ]
    assert main([*argv, "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["audits"][0]["acceptance_contract"]["status"] == "error"
    assert main(argv) == 1
    human = capsys.readouterr().out
    assert "Required BODY order" in human and "left_foot" in human
    assert "right_foot" in human and "Reference" in human and "Candidate" in human
    assert '"reference_body"' not in human  # Human result is a concise comparison row.
    assert main([*argv, "--markdown"]) == 1
    markdown = capsys.readouterr().out
    assert "| Component | Reference | Candidate |" in markdown
    case = tmp_path / "case"
    assert (
        main(
            [
                "create-case",
                "resolved-layout",
                str(layouts["before"]),
                str(case),
                "--corrected",
                str(layouts["corrected"]),
                "--requirements",
                str(path),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["reproduce-case", str(case), "--json"]) == 0
    capsys.readouterr()
    assert main(["check-case", str(case), str(layouts["later"]), "--json"]) == 0
    capsys.readouterr()


def test_existing_mcp_same_requirements_and_both_root_guards(layouts, tmp_path):
    pytest.importorskip("mcp")
    from policy_doctor.server import create_server

    tools = create_server([str(tmp_path)])._tool_manager._tools
    assert len(tools) == 29
    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
    fn = tools["investigate_run_directory"].fn
    report = fn(
        str(layouts["before"]), reference=str(layouts["corrected"]), requirements=layout_requirement()
    )
    assert report["status"] == "error"
    with pytest.raises(InputError, match="outside"):
        fn(str(layouts["before"]), reference="/outside", requirements=layout_requirement())
    with pytest.raises(InputError):
        fn(str(layouts["before"]), requirements=layout_requirement(atol=True))


def test_explicit_named_tolerance_is_used_for_audit_and_reference_comparison(tmp_path, monkeypatch):
    from test_named_actions import change

    packet = packet_fixture.__wrapped__(tmp_path, monkeypatch)
    before, after = packet("before", limited=False), packet("after", limited=False)

    def subtle(data):
        data["ticks"][0]["driver_return"]["a.pos"] += 0.0000001
        data["ticks"][0]["wrapper_return"]["a.pos"] += 0.0000001

    change(before, subtle)
    requirement = {
        "kind": "native-named-action",
        "parameters": {"require_canonical_matches_driver_return": True, "atol": 0.0, "rtol": 0.0},
    }
    result = investigate_run(str(before), reference=str(after), requirements=requirement)
    assert result["status"] == "error"
    first = result["comparison"]["first_supported_difference"]
    assert first["stage"] == "driver_return"
    assert first["candidate"]["a.pos"] != first["reference"]["a.pos"]
    assert result["case_recipe"]["arguments"]["parameters"]["atol"] == 0.0


def test_cli_conflicting_old_flags_refuse_before_any_case_write(layouts, tmp_path, capsys):
    from policy_doctor.cli import main

    path = tmp_path / "requirements.json"
    path.write_text(json.dumps(layout_requirement()))
    destination = tmp_path / "must-not-exist"
    assert (
        main(
            [
                "create-case",
                "resolved-layout",
                str(layouts["before"]),
                str(destination),
                "--corrected",
                str(layouts["corrected"]),
                "--requirements",
                str(path),
                "--expected-names",
                "left_foot",
                "right_foot",
                "--json",
            ]
        )
        == 2
    )
    assert not destination.exists()
    assert "together" in capsys.readouterr().err
    assert (
        main(
            [
                "create-case",
                "native-named-action",
                str(layouts["before"]),
                str(destination),
                "--corrected",
                str(layouts["corrected"]),
                "--requirements",
                str(path),
                "--json",
            ]
        )
        == 2
    )
    assert not destination.exists()
    capsys.readouterr()


def test_unknown_namespace_prevents_a_recipe_even_when_body_requirement_passes(layouts):
    report = investigate_run(
        str(layouts["before"]), reference=str(layouts["unknown"]), requirements=layout_requirement()
    )
    assert report["comparison"]["reference_criterion_status"] == "pass"
    assert report["case_recipe"] is None


def test_raw_mcp_tolerances_and_required_params_are_not_dropped(layouts, tmp_path):
    pytest.importorskip("mcp")
    from policy_doctor.server import create_server

    tools = create_server([str(tmp_path)])._tool_manager._tools
    report = tools["investigate_run_directory"].fn(
        str(layouts["before"]),
        reference=str(layouts["corrected"]),
        requirements=layout_requirement(atol=0.0, rtol=0.0),
    )
    assert report["audits"][0]["tolerances"] == {"atol": 0.0, "rtol": 0.0}
    action = next(action for action in report["next_actions"] if action["tool"] == "audit_runtime_capture")
    assert action["arguments"]["atol"] == 0.0 and action["arguments"]["rtol"] == 0.0
    assert tools["audit_runtime_capture"].fn(**action["arguments"])["status"] == "error"
    with pytest.raises(InputError):
        tools["audit_runtime_capture"].fn(str(layouts["before"]), atol=True)


def test_source_input_and_requirement_bytes_unchanged_after_investigation(layouts, tmp_path):
    from hashlib import sha256

    files = {
        path: sha256(path.read_bytes()).hexdigest()
        for root in layouts.values()
        for path in root.rglob("*")
        if path.is_file()
    }
    investigate_run(
        str(layouts["before"]), reference=str(layouts["corrected"]), requirements=layout_requirement()
    )
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in files.items())


@pytest.mark.parametrize("kind", ["resolved-layout", "native-named-action"])
def test_disclosed_full_recipe_and_same_file_creation_match_manifest_exactly(
    kind, layouts, tmp_path, monkeypatch, capsys
):
    from policy_doctor.cli import main

    if kind == "resolved-layout":
        before, after = layouts["before"], layouts["corrected"]
        requirement = layout_requirement()
        inferred_keys = {"expected_frame", "expected_quantity", "expected_identity"}
    else:
        packet = packet_fixture.__wrapped__(tmp_path, monkeypatch)
        before, after = packet("named-before"), packet("named-after", limited=False)
        requirement = {"kind": kind, "parameters": {"require_canonical_matches_driver_return": True}}
        inferred_keys = {"expected_action_names", "expected_driver_namespace"}
    result = investigate_run(str(before), reference=str(after), requirements=requirement)
    recipe = result["case_recipe"]
    assert set(recipe["retained_context"]) == inferred_keys
    assert recipe["supplied_requirements"] == result["applied_requirements"] == requirement
    assert all(recipe["parameter_origins"][key] == "corrected_source_context" for key in inferred_keys)
    path = tmp_path / ("requirements-" + kind + ".json")
    path.write_text(json.dumps(requirement))
    case = tmp_path / ("case-" + kind)
    assert (
        main(
            [
                "create-case",
                kind,
                str(before),
                str(case),
                "--corrected",
                str(after),
                "--requirements",
                str(path),
                "--json",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["retained_context"] == recipe["retained_context"]
    manifest = json.loads((case / "manifest.json").read_text())
    assert manifest["parameters"] == recipe["arguments"]["parameters"]


def test_advertised_standalone_named_audit_markdown_has_scoped_evidence(tmp_path, monkeypatch, capsys):
    from policy_doctor.cli import main
    packet = packet_fixture.__wrapped__(tmp_path, monkeypatch)
    root = packet('named')
    assert main(['audit-capture', str(root), '--markdown']) == 3
    rendered = capsys.readouterr().out
    assert 'Named software boundary' in rendered
    assert 'driver changes 1' in rendered
    assert 'Physical execution' in rendered
    assert 'SHA-256' in rendered
