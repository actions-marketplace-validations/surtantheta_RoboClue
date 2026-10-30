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


def test_workflow_has_manual_and_weekly_review_without_publication():
    import yaml

    path = Path(__file__).parents[1] / ".github/workflows/upstream-revalidation.yml"
    document = yaml.safe_load(path.read_text())
    assert "workflow_dispatch" in document["on"] and document["on"]["schedule"]
    assert document["permissions"] == {"contents": "read"}
    text = path.read_text()
    assert "native GPU" in text and "issues: write" not in text and "pull-requests: write" not in text
