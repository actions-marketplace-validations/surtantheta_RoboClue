"""Independent N1/N4 acceptance: generated controls, no live network."""

import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor import hub
from policy_doctor.artifacts import InputError
from policy_doctor.demo import create_policy_demo
from policy_doctor.server import create_server


@pytest.fixture
def demo(tmp_path):
    return create_policy_demo(tmp_path / "generated")


def canonical(report):
    return json.dumps(report, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@pytest.fixture
def generated_hub(monkeypatch):
    features = {"observation.state": {"type": "STATE", "shape": [2]},
                "action": {"type": "ACTION", "shape": [2]}}
    normalization = {"features": features, "norm_map": {"STATE": "IDENTITY", "ACTION": "IDENTITY"}}
    files = {"model": {
        "config.json": {"type": "act", "input_features": {"observation.state": features["observation.state"]},
                        "output_features": {"action": features["action"]}},
        "policy_preprocessor.json": {"steps": [{"registry_name": "normalizer_processor", "config": normalization}]},
        "policy_postprocessor.json": {"steps": [{"registry_name": "unnormalizer_processor", "config": normalization}]}},
        "data": {"meta/info.json": {"fps": 30, "features": {
            key: dict(value, names=["shoulder", "gripper"]) for key, value in features.items()}}}}
    # Separate files must not share mutable dict objects in this fixture.
    files = json.loads(json.dumps(files))
    calls = []

    def request(url, *, limit, required_sha=None):
        calls.append((url, required_sha))
        if "/api/models/" in url:
            repo, sha = "model", "1" * 40
        elif "/api/datasets/" in url:
            repo, sha = "data", "2" * 40
        else:
            repo = "data" if "/datasets/" in url else "model"
            sha = ("2" if repo == "data" else "1") * 40
        if "/api/" in url:
            data = {"id": "org/" + repo, "sha": sha, "private": False,
                    "siblings": [{"rfilename": name} for name in files[repo]]}
        else:
            assert required_sha == sha
            key = url.split(f"/resolve/{sha}/")[1]
            data = files[repo][key]
        return json.dumps(data).encode()

    monkeypatch.setattr(hub, "_request", request)
    return files, calls


def test_metadata_scope_excludes_exactly_the_original_missing_deployment(demo):
    args = (demo["checkpoint"], demo["corrected_dataset"])
    default = hub.check_checkpoint(*args)
    explicit = hub.check_checkpoint(*args, scope="deployment")
    metadata = hub.check_checkpoint(*args, scope="metadata")
    assert default == explicit
    excluded = [f for f in default["findings"] if f["code"] == "deployment.missing"]
    assert len(excluded) == 1 and excluded[0]["status"] == "unknown"
    assert metadata["unassessed_findings"] == excluded
    assert default["unassessed_findings"] == []
    assert metadata["findings"] == [f for f in default["findings"] if f not in excluded]
    assert metadata["artifacts"] == default["artifacts"]
    assert metadata["status"] == "pass" and default["status"] == "unknown"
    assert metadata["assessment_scope"] == "metadata"
    assert metadata["deployment_status"] == default["deployment_status"] == "not_assessed"
    assert metadata["counts"]["unknown"] == 0
    assert metadata["counts"]["pass"] == default["counts"]["pass"]


@pytest.mark.parametrize("condition", ["contradiction", "custom_processor"])
def test_metadata_scope_does_not_suppress_real_error_or_unknown(demo, condition):
    dataset = demo["dataset"] if condition == "contradiction" else demo["corrected_dataset"]
    if condition == "custom_processor":
        path = Path(demo["checkpoint"]) / "policy_preprocessor.json"
        data = json.loads(path.read_text())
        data["steps"].append({"registry_name": "new_unmodeled_transform", "config": {}})
        path.write_text(json.dumps(data))
    before = hub.check_checkpoint(demo["checkpoint"], dataset)
    report = hub.check_checkpoint(demo["checkpoint"], dataset, scope="metadata")
    assert report["status"] == ("error" if condition == "contradiction" else "unknown")
    assert report["findings"] == [f for f in before["findings"] if f["code"] != "deployment.missing"]
    assert any(f["status"] == report["status"] for f in report["findings"])


@pytest.mark.parametrize("scope", ["all", "metadata-only", None, False, {}])
def test_invalid_scope_refuses_before_any_intake(monkeypatch, scope):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Invalid scope must fail before intake")

    monkeypatch.setattr(hub, "_request", forbidden)
    monkeypatch.setattr(hub, "audit", forbidden)
    with pytest.raises(InputError, match="scope"):
        hub.check_checkpoint("hf://org/model", fetch=True, scope=scope)
    assert calls == []


def test_scope_cli_healthy_pass_and_default_unknown(demo, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    args = [sys.executable, "-m", "policy_doctor", "check", demo["checkpoint"],
            "--dataset", demo["corrected_dataset"], "--json"]
    default = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    scoped = subprocess.run([*args, "--scope", "metadata"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert default.returncode == 3, default.stderr
    assert scoped.returncode == 0, scoped.stderr
    body = json.loads(scoped.stdout)
    assert body["status"] == "pass" and body["deployment_status"] == "not_assessed"
    assert [f["code"] for f in body["unassessed_findings"]] == ["deployment.missing"]


def test_concise_retains_verdict_counts_and_a_complete_report_digest(demo):
    from policy_doctor.check_report import concise_check_summary

    source, dataset = demo["checkpoint"], demo["dataset"]
    report = hub.check_checkpoint(source, dataset, scope="metadata")
    original = copy.deepcopy(report)
    summary = concise_check_summary(report, source=source, dataset=dataset, scope="metadata")
    assert report == original
    assert summary["status"] == report["status"] == "error"
    assert summary["counts"] == report["counts"]
    assert summary["assessment_scope"] == "metadata"
    assert summary["deployment_status"] == "not_assessed"
    assert summary["full_report"]["sha256"] == hashlib.sha256(canonical(report)).hexdigest()
    assert summary["full_report"]["size_bytes"] == len(canonical(report))
    assert len(canonical(summary)) <= 8192
    assert summary["shown_findings"] + summary["omitted_findings"] == summary["finding_count"]
    assert summary["finding_count"] == len(report["findings"])
    arguments = summary["detailed_request"]["arguments"]
    assert arguments["source"] == source and arguments["dataset"] == dataset
    assert arguments["scope"] == "metadata" and arguments["response_format"] == "detailed"


def test_concise_large_unicode_report_exposes_error_and_omissions(demo):
    from policy_doctor.check_report import concise_check_summary

    report = hub.check_checkpoint(demo["checkpoint"], demo["corrected_dataset"], scope="metadata")
    witness = {"path": "/generated/metadata.json", "sha256": "f" * 64,
               "pointer": "/generated", "value": "机械关节" * 2000}
    report["findings"] = [{"code": f"generated.unknown.{n}", "status": "unknown",
                           "message": "Generated unmodeled boundary " * 100,
                           "evidence": [copy.deepcopy(witness) for _ in range(5)]}
                          for n in range(20)]
    report["findings"].append({"code": "generated.error.last", "status": "error",
                               "message": "The final error must remain visible", "evidence": [witness]})
    report["status"] = "error"
    report["counts"] = {"error": 1, "warning": 0, "unknown": 20, "pass": 0}
    original_sha = hashlib.sha256(canonical(report)).hexdigest()
    summary = concise_check_summary(report, source=demo["checkpoint"],
                                    dataset=demo["corrected_dataset"], scope="metadata")
    assert len(json.dumps(summary).encode()) <= 8192
    assert summary["status"] == "error" and summary["counts"] == report["counts"]
    assert summary["full_report"]["sha256"] == original_sha
    assert summary["shown_findings"] <= 3
    assert summary["shown_findings"] + summary["omitted_findings"] == 21
    assert summary["omitted_findings"] > 0 and summary["detail_complete"] is False
    assert any(f["code"] == "generated.error.last" for f in summary["findings"])
    for finding in summary["findings"]:
        assert len(finding["evidence"]) + finding["omitted_evidence"] == finding["evidence_count"]
        for evidence in finding["evidence"]:
            assert evidence["path"] == witness["path"]
            assert evidence["pointer"] == witness["pointer"]
            assert evidence["sha256"] == witness["sha256"]
            assert evidence["value_preview"]["truncated"] is True


def test_mcp_default_catalog_is_offline_and_scope_format_are_typed(tmp_path):
    async def run():
        default = create_server([str(tmp_path)])
        assert "check_policy_metadata" not in {t.name for t in await default.list_tools()}
        enabled = create_server([str(tmp_path)], allow_hub=True)
        tool = next(t for t in await enabled.list_tools() if t.name == "check_policy_metadata")
        fields = tool.inputSchema["properties"]
        assert fields["scope"]["enum"] == ["deployment", "metadata"]
        assert fields["scope"]["default"] == "deployment"
        assert fields["response_format"]["enum"] == ["concise", "detailed"]
        assert fields["response_format"]["default"] == "detailed"

    asyncio.run(run())


def test_source_stdio_scope_and_concise_match_the_original_detailed_report(demo, tmp_path):
    async def run():
        parameters = StdioServerParameters(command=sys.executable,
            args=["-m", "policy_doctor", "serve", "--root", str(tmp_path), "--allow-hub"],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")))
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                async def call(arguments):
                    reply = await session.call_tool("check_policy_metadata", arguments)
                    assert not reply.isError
                    return reply.structuredContent or json.loads(reply.content[0].text)

                for scope, expected in (("deployment", "unknown"), ("metadata", "pass")):
                    args = {"source": demo["checkpoint"], "dataset": demo["corrected_dataset"], "scope": scope}
                    detailed = await call(args)  # Original default remains detailed.
                    concise = await call(dict(args, response_format="concise"))
                    assert detailed["kind"] == "checkpoint_check"
                    assert concise["kind"] == "checkpoint_check_summary"
                    assert concise["status"] == detailed["status"] == expected
                    assert concise["counts"] == detailed["counts"]
                    assert concise["deployment_status"] == "not_assessed"
                    assert concise["assessment_scope"] == scope
                    assert len(json.dumps(concise).encode()) <= 8192
                    assert concise["full_report"]["sha256"] == hashlib.sha256(canonical(detailed)).hexdigest()

    asyncio.run(asyncio.wait_for(run(), timeout=30))


def test_concise_hub_drilldown_pins_revisions_and_preserves_checked_evidence(generated_hub):
    from policy_doctor.check_report import concise_check_summary

    _, calls = generated_hub
    report = hub.check_checkpoint("hf://org/model", "hf://org/data", fetch=True, scope="metadata")
    summary = concise_check_summary(report, source="hf://org/model", dataset="hf://org/data", scope="metadata")
    arguments = dict(summary["detailed_request"]["arguments"])
    assert arguments.pop("response_format") == "detailed"
    assert arguments["revision"] == "1" * 40
    assert arguments["dataset_revision"] == "2" * 40
    assert arguments["scope"] == "metadata"
    replayed = hub.check_checkpoint(**arguments, fetch=True)
    assert replayed["findings"] == report["findings"]
    assert replayed["status"] == report["status"] and replayed["counts"] == report["counts"]
    assert replayed["artifacts"] == report["artifacts"]
    for old, new in zip(report["provenance"], replayed["provenance"], strict=True):
        assert old["resolved_revision"] == new["resolved_revision"]
        assert old["files"] == new["files"]
        assert old["requested_revision"] == "main"
        assert new["requested_revision"] == new["resolved_revision"]
    assert hashlib.sha256(canonical(report)).hexdigest() == summary["full_report"]["sha256"]
    assert "new read" in summary["detailed_request_note"].lower()
    assert all(required is not None for url, required in calls if "/resolve/" in url)


def test_metadata_scope_keeps_positively_listed_uninspected_state_unknown(generated_hub):
    files, _ = generated_hub
    step = files["model"]["policy_preprocessor.json"]["steps"][0]
    step["config"]["norm_map"] = {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    step["state_file"] = "pre.safetensors"
    files["model"]["pre.safetensors"] = {"never": "fetch"}
    post = files["model"]["policy_postprocessor.json"]["steps"][0]
    post["config"]["norm_map"] = {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    post["state_file"] = "post.safetensors"
    files["model"]["post.safetensors"] = {"never": "fetch"}
    result = hub.check_checkpoint("hf://org/model", "hf://org/data", fetch=True, scope="metadata")
    assert result["status"] == "unknown"
    assert any(f["code"] == "processor.state_uninspected" for f in result["findings"])
    assert not any(f["code"] == "processor.state_missing" for f in result["findings"])


@pytest.mark.parametrize("format_value", ["short", False, None])
def test_invalid_summary_format_and_outside_roots_refuse_before_network(tmp_path, monkeypatch, format_value):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("No request before schema and read-root authorization")

    monkeypatch.setattr(hub, "_request", forbidden)

    async def run():
        server = create_server([str(tmp_path)], allow_hub=True)
        with pytest.raises(Exception, match="response_format"):
            await server.call_tool("check_policy_metadata", {
                "source": "hf://org/model", "response_format": format_value})
        with pytest.raises(Exception, match="outside"):
            await server.call_tool("check_policy_metadata", {
                "source": "hf://org/model", "dataset": "/etc", "response_format": "concise"})

    asyncio.run(run())
    assert calls == []
