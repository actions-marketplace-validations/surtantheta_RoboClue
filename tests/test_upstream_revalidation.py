"""Observed source drift cannot turn an untested release into qualified support."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("roboclue_upstream_probe", Path(__file__).parents[1] / "scripts/revalidate_upstream.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _profile():
    return {"id": "example", "repository": "example/framework", "tested_commits": ["a" * 40], "rule_ids": ["RC001"]}


@pytest.mark.parametrize("head, expected", [("a" * 40, "matches_recorded_source_pin"), ("b" * 40, "source_changed_requires_revalidation")])
def test_head_comparison_is_scoped_to_source_identity(monkeypatch, head, expected):
    monkeypatch.setattr(probe, "fetch_json", lambda url: {"sha": head} if url.endswith("HEAD") else {"tag_name": "v999.0"})
    result = probe.observe(_profile())
    assert result["status"] == expected and result["qualification"] == "not_rerun"
    assert result["latest_release_tag"] == "v999.0"
    assert "not qualified" in result["release_scope"]


@pytest.mark.parametrize("response", [{}, {"sha": "main"}, {"sha": "A" * 40}, {"sha": []}])
def test_missing_or_malformed_source_evidence_remains_unknown(monkeypatch, response):
    monkeypatch.setattr(probe, "fetch_json", lambda _url: response)
    assert probe.observe(_profile())["status"] == "unknown"


def test_network_error_remains_unknown_without_exposing_authorization(monkeypatch):
    def fail(_url):
        raise OSError("private credential must not appear")

    monkeypatch.setattr(probe, "fetch_json", fail)
    result = probe.observe(_profile())
    assert result["status"] == "unknown" and "credential" not in json.dumps(result)


def test_invalid_catalog_repository_is_not_requested(monkeypatch):
    monkeypatch.setattr(probe, "fetch_json", lambda *_: pytest.fail("invalid repository was fetched"))
    with pytest.raises(ValueError):
        probe.observe(dict(_profile(), repository="example/framework/../../other"))


def test_authorization_never_follows_a_cross_host_redirect():
    request = probe.urllib.request.Request("https://api.github.com/repos/example/framework", headers={"Authorization": "Bearer test-only"})
    with pytest.raises(ValueError, match="authorized GitHub API host"):
        probe._GitHubRedirect().redirect_request(request, None, 302, "redirect", {}, "https://other.example/collect")


def test_main_retains_changed_source_as_unqualified_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "fetch_json", lambda url: {"sha": "b" * 40} if url.endswith("HEAD") else {"tag_name": "v999"})
    output = tmp_path / "observation.json"
    assert probe.main(["--fetch", "--output", str(output)]) == 3
    result = json.loads(output.read_text())
    assert result["native_framework_qualification"] == "not_rerun"
    assert result["status"] == "revalidation_required" and len(result["frameworks"]) == 4
    assert all(item["status"] == "source_changed_requires_revalidation" for item in result["frameworks"])
    before = output.read_bytes()
    assert probe.main(["--fetch", "--output", str(output)]) == 2
    assert output.read_bytes() == before


def test_fetch_permission_is_required_before_observations(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "fetch_json", lambda *_: pytest.fail("network intake without --fetch"))
    assert probe.main(["--output", str(tmp_path / "none.json")]) == 2
    assert not (tmp_path / "none.json").exists()


@pytest.mark.parametrize("github_actions", [False, True])
@pytest.mark.parametrize("head, expected", [("a" * 40, "recorded_pins_observed"), ("b" * 40, "revalidation_required"), (None, "revalidation_required")])
def test_ci_reports_drift_without_hiding_missing_evidence(monkeypatch, tmp_path, capsys, github_actions, head, expected):
    from policy_doctor import framework_versions

    monkeypatch.setattr(framework_versions, "supported_framework_profiles", lambda: {"frameworks": [_profile()]})
    monkeypatch.setattr(probe, "fetch_json", lambda url: {"sha": head} if url.endswith("HEAD") else {"tag_name": "v999"})
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    output = tmp_path / "observation.json"
    args = ["--fetch", "--output", str(output)] + (["--github-actions"] if github_actions else [])
    code = probe.main(args)
    assert code == ((2 if head is None else 0) if github_actions else (0 if head == "a" * 40 else 3))
    result = json.loads(output.read_text())
    assert result["status"] == expected
    assert result["native_framework_qualification"] == "not_rerun"
    assert result["frameworks"][0]["recorded_tested_commits"] == ["a" * 40]
    logs = capsys.readouterr().out
    if not github_actions:
        assert not summary.exists()
        assert "::warning" not in logs and "::error" not in logs
    else:
        text = summary.read_text()
        assert "Native qualification was not rerun" in text
        if head is None:
            assert "::error title=Upstream observation unavailable::" in logs
            assert "Unavailable" in text
        elif head == "b" * 40:
            assert "::warning title=Upstream review needed::" in logs
            assert "Changed; qualification pending" in text
            assert f"https://github.com/example/framework/compare/{'a' * 40}...{'b' * 40}" in text
        else:
            assert "::warning" not in logs and "::error" not in logs
            assert "Matches a recorded source pin" in text


def test_ci_mixed_drift_and_network_failure_is_not_success(monkeypatch, tmp_path, capsys):
    from policy_doctor import framework_versions

    profiles = [_profile(), dict(_profile(), id="unavailable", repository="example/unavailable")]
    monkeypatch.setattr(framework_versions, "supported_framework_profiles", lambda: {"frameworks": profiles})

    def fetch(url):
        if "example/unavailable" in url:
            raise OSError("secret must not appear")
        return {"sha": "b" * 40} if url.endswith("HEAD") else {"tag_name": "v999"}

    monkeypatch.setattr(probe, "fetch_json", fetch)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    output = tmp_path / "observation.json"
    assert probe.main(["--fetch", "--github-actions", "--output", str(output)]) == 2
    logs = capsys.readouterr().out
    assert "::warning" in logs and "::error" in logs
    assert "secret" not in logs + output.read_text() + summary.read_text()


def test_ci_requires_summary_destination_before_fetch(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(probe, "fetch_json", lambda *_: pytest.fail("network intake without summary"))
    assert probe.main(["--fetch", "--github-actions", "--output", str(tmp_path / "none.json")]) == 2
    assert not (tmp_path / "none.json").exists()


def test_ci_summary_write_failure_is_not_success(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "fetch_json", lambda url: {"sha": "b" * 40} if url.endswith("HEAD") else {"tag_name": "v999"})
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "missing" / "summary.md"))
    assert probe.main(["--fetch", "--github-actions", "--output", str(tmp_path / "observation.json")]) == 2


def test_empty_profile_catalog_is_not_success(monkeypatch, tmp_path):
    from policy_doctor import framework_versions

    monkeypatch.setattr(framework_versions, "supported_framework_profiles", lambda: {"frameworks": []})
    assert probe.main(["--fetch", "--output", str(tmp_path / "none.json")]) == 2
    assert not (tmp_path / "none.json").exists()


def test_workflow_has_manual_and_weekly_review_without_publication():
    import yaml

    path = Path(__file__).parents[1] / ".github/workflows/upstream-revalidation.yml"
    document = yaml.safe_load(path.read_text())
    assert "workflow_dispatch" in document["on"] and document["on"]["schedule"]
    assert document["permissions"] == {"contents": "read"}
    assert "pull_request" in document["on"]
    tests = document["jobs"]["portable-regressions"]
    monitor = document["jobs"]["upstream-source-monitor"]
    assert "needs" not in tests and "needs" not in monitor
    assert any("pytest" in step.get("run", "") for step in tests["steps"])
    assert any("--github-actions" in step.get("run", "") for step in monitor["steps"])
    assert all(not step.get("continue-on-error") for job in document["jobs"].values() for step in job["steps"])
    text = path.read_text()
    assert "native GPU" in text and "issues: write" not in text and "pull-requests: write" not in text
