"""The natural branded recovery command retains the existing checker's contract."""

import json

import pytest

from policy_doctor.big_bets_frontdoor import main as branded_main
from policy_doctor.cli import main as original_main
from policy_doctor.server import create_server


def test_branded_preflight_retains_narrowed_coverage_and_original_result(tmp_path, capsys):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v2.1", "fps": 30,
        "features": {"action": {"dtype": "float32", "shape": [1], "names": ["joint"]}}}))
    args = ["preflight", str(root), "--no-groot-contract", "--json"]
    expected_exit = original_main(args)
    original = json.loads(capsys.readouterr().out)
    assert expected_exit == 3 and original["status"] == "unknown"
    actual_exit = branded_main(args)
    actual = json.loads(capsys.readouterr().out)
    assert actual_exit == expected_exit and actual == original
    assert actual["coverage"]["groot_contract_skipped"]


@pytest.mark.parametrize("main", [branded_main, original_main])
def test_branded_preflight_preserves_unknown_argument_refusal(main):
    with pytest.raises(SystemExit) as error:
        main(["preflight", ".", "--invent-health"])
    assert error.value.code == 2


def test_mcp_generic_card_schema_requires_explicit_groot_contract(tmp_path):
    tools = create_server([str(tmp_path)])._tool_manager._tools
    assert tools["dataset_health_card"].parameters["properties"]["groot_contract"]["default"] is False
