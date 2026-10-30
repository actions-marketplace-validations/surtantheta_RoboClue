"""A broken artifact must not erase independent CI checks or their receipts."""
import json
import os

import pytest

from policy_doctor import automation


def _write(path, data):
    path.write_text(json.dumps(data))


def _manifest(root, checks):
    _write(root / "checks.json", {"schema_version": 1, "checks": checks})
    return "checks.json"


@pytest.fixture
def consumer(tmp_path):
    for name, value in (("reviewed.json", {"camera": "front"}), ("healthy.json", {"camera": "front"}),
                        ("changed.json", {"camera": "wrist"})):
        _write(tmp_path / name, value)
    return tmp_path


def _check(name, artifact, **options):
    return dict(id=name, check="config", artifact=artifact, reference="reviewed.json", **options)


def test_missing_first_artifact_keeps_later_fault_and_healthy_results(consumer):
    path = _manifest(consumer, [_check("missing", "does-not-exist.json"), _check("fault", "changed.json"),
                                _check("healthy", "healthy.json")])
    out = consumer / "receipts"
    result = automation.run_manifest(path, workspace=consumer, output=out)
    assert result["exit_code"] == 2 and result["gate"] == "blocked"
    assert [check["exit_code"] for check in result["checks"]] == [2, 1, 0]
    assert result["checks"][0]["error"]["code"] == "invalid_input"
    assert "Not found" in result["checks"][0]["error"]["message"]
    assert {p.name for p in out.iterdir()} == {"missing.json", "fault.json", "healthy.json", "result.json"}
    assert json.loads((out / "missing.json").read_text())["report"]["error"]["code"] == "invalid_input"
    assert json.loads((out / "fault.json").read_text())["report"]["equal"] is False
    assert json.loads((out / "healthy.json").read_text())["report"]["equal"] is True


@pytest.mark.parametrize("kind", ["missing-reference", "bad-json", "fifo", "symlink"])
def test_invalid_check_inputs_are_named_and_do_not_abort_other_checks(consumer, kind):
    bad = _check("invalid", "changed.json")
    if kind == "missing-reference":
        bad["reference"] = "missing.json"
    elif kind == "bad-json":
        (consumer / "changed.json").write_text('{"truncated":')
    elif kind == "fifo":
        os.mkfifo(consumer / "special.json")
        bad["artifact"] = "special.json"
    else:
        (consumer / "linked.json").symlink_to(consumer / "changed.json")
        bad["artifact"] = "linked.json"
    path = _manifest(consumer, [bad, _check("healthy", "healthy.json")])
    result = automation.run_manifest(path, workspace=consumer, output=consumer / "receipts")
    assert [r["exit_code"] for r in result["checks"]] == [2, 0]
    assert result["checks"][0]["error"]["message"]


def test_precommit_oversized_check_keeps_an_independent_small_comparison(consumer):
    (consumer / "large.json").write_text('{"data":"' + "x" * automation.PRECOMMIT_FILE_LIMIT + '"}')
    path = _manifest(consumer, [_check("large", "large.json"), _check("healthy", "healthy.json")])
    result = automation.run_manifest(path, workspace=consumer, filenames=[], output=consumer / "receipts")
    assert [r["exit_code"] for r in result["checks"]] == [2, 0]


def test_invalid_whole_manifest_writes_global_receipt_without_running_any_check(consumer, monkeypatch):
    bad = _check("bad", "changed.json")
    bad["shell"] = "never execute this"
    path = _manifest(consumer, [_check("healthy", "healthy.json"), bad])
    from policy_doctor import product_cli

    monkeypatch.setattr(product_cli, "main", lambda *_: pytest.fail("invalid schema ran a check"))
    out = consumer / "receipts"
    assert automation.main(["run", "--workspace", str(consumer), "--manifest", path, "--output", str(out)]) == 2
    result = json.loads((out / "result.json").read_text())
    assert result["error"]["code"] == "invalid_manifest" and result["checks"] == []
    assert result["exit_code"] == 2 and result["gate"] == "blocked"
    assert {p.name for p in out.iterdir()} == {"result.json"}


@pytest.mark.parametrize("kind", ["missing-manifest", "malformed-manifest"])
def test_unreadable_manifest_still_retains_safe_requested_receipt(consumer, kind):
    if kind == "malformed-manifest":
        (consumer / "checks.json").write_text('{"bad":')
    out = consumer / "receipts"
    assert automation.main(["run", "--workspace", str(consumer), "--manifest", "checks.json", "--output", str(out)]) == 2
    result = json.loads((out / "result.json").read_text())
    assert result["checks"] == [] and result["exit_code"] == 2


def test_invalid_schema_does_not_write_error_receipt_inside_named_input_directory(consumer):
    (consumer / "dataset").mkdir()
    path = _manifest(consumer, [{"id": "data", "check": "dataset", "artifact": "dataset", "unknown_key": True}])
    out = consumer / "dataset/receipts"
    assert automation.main(["run", "--workspace", str(consumer), "--manifest", path, "--output", str(out)]) == 2
    assert not out.exists()


def test_invalid_manifest_never_overwrites_an_existing_receipt_or_follows_a_link(consumer):
    (consumer / "checks.json").write_text('{"bad":')
    out = consumer / "existing"
    out.mkdir()
    (out / "result.json").write_text("preserve")
    alias = consumer / "alias"
    alias.symlink_to(out, target_is_directory=True)
    for destination in (out, alias / "new"):
        assert automation.main(["run", "--workspace", str(consumer), "--manifest", "checks.json", "--output", str(destination)]) == 2
    assert (out / "result.json").read_text() == "preserve" and not (out / "new").exists()


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_checker_input_error_protocol_is_retained_from_either_stream(consumer, monkeypatch, stream):
    from policy_doctor import product_cli

    actual = product_cli.main
    first = True

    def check(argv):
        nonlocal first
        if first:
            import sys

            first = False
            print(json.dumps({"error": {"code": "invalid_input", "message": "named input failure"}}),
                  file=getattr(sys, stream))
            return 2
        return actual(argv)

    monkeypatch.setattr(product_cli, "main", check)
    path = _manifest(consumer, [_check("invalid", "changed.json"), _check("healthy", "healthy.json")])
    result = automation.run_manifest(path, workspace=consumer)
    assert [r["exit_code"] for r in result["checks"]] == [2, 0]
    assert result["checks"][0]["error"]["message"] == "named input failure"


def test_argparse_exit_is_a_per_check_error_not_an_aggregate_abort(consumer, monkeypatch):
    from policy_doctor import product_cli

    actual = product_cli.main
    first = True

    def check(argv):
        nonlocal first
        if first:
            first = False
            raise SystemExit(2)
        return actual(argv)

    monkeypatch.setattr(product_cli, "main", check)
    path = _manifest(consumer, [_check("invalid", "changed.json"), _check("healthy", "healthy.json")])
    result = automation.run_manifest(path, workspace=consumer)
    assert [r["exit_code"] for r in result["checks"]] == [2, 0]
