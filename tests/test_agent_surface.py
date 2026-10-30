"""The agent-facing surface stays in step with the engine: skill coverage, schemas, prompts and limits."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from policy_doctor.incidents import CHECKS
from policy_doctor.server import RESULT_LIMIT, _summary, create_server

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents" / "skills" / "robot-incident-investigation" / "SKILL.md"


def test_skill_names_every_retained_case_kind():
    text = SKILL.read_text()
    missing = [kind for kind in sorted(CHECKS) if f"`{kind.replace('_', '-')}`" not in text]
    assert not missing, f"SKILL.md does not describe case kinds: {missing}"


def _tools(server):
    return {tool.name: tool for tool in asyncio.run(server.list_tools())}


def test_no_untyped_tool_parameters(tmp_path):
    tools = _tools(create_server([str(tmp_path)], case_root=str(tmp_path), allow_hub=True, allow_hub_media=True))
    untyped = [(name, param) for name, tool in tools.items()
               for param, schema in tool.inputSchema.get("properties", {}).items()
               if not any(key in schema for key in ("type", "anyOf", "oneOf", "enum", "$ref"))]
    assert not untyped
    schema = tools['recover_dataset_video_reference'].inputSchema['properties']['episode_index']
    assert schema['type'] == 'integer' and schema['minimum'] == 0


def test_boolean_tolerance_still_reaches_the_engine_guard(tmp_path):
    # Typed schemas must not let the SDK coerce rtol=true into 1.0.
    server = create_server([str(tmp_path)])
    (tmp_path / "a.npz").write_bytes(b"")

    async def call():
        return await server.call_tool("compare_rollout_recordings",
                                      {"reference": str(tmp_path / "a.npz"), "candidate": str(tmp_path / "a.npz"), "rtol": True})

    try:
        asyncio.run(call())
        raise AssertionError("boolean tolerance accepted")
    except Exception as exc:  # FastMCP wraps the engine's InputError
        assert "tolerance" in str(exc).lower() or "rows" in str(exc).lower() or "recording" in str(exc).lower()


def test_prompts_and_guide_resource(tmp_path):
    server = create_server([str(tmp_path)], allow_hub=True)
    prompts = {p.name for p in asyncio.run(server.list_prompts())}
    assert {"investigate_robot_run", "check_public_policy"} <= prompts
    assert "check_public_policy" not in {p.name for p in asyncio.run(create_server([str(tmp_path)]).list_prompts())}
    guide = asyncio.run(server.read_resource("roboclue://guide"))
    text = "".join(getattr(part, "content", "") for part in guide)
    assert str(tmp_path.resolve()) in text and all(kind in text for kind in CHECKS)


def test_out_of_root_error_names_the_roots(tmp_path):
    server = create_server([str(tmp_path)])
    try:
        asyncio.run(server.call_tool("inspect_checkpoint", {"checkpoint": "/etc"}))
        raise AssertionError("escape accepted")
    except Exception as exc:
        assert str(tmp_path.resolve()) in str(exc)


def test_summary_keeps_verdict_and_attention_findings():
    findings = [{"code": f"c{i}", "status": "pass", "message": "ok", "evidence": []} for i in range(5000)]
    findings.insert(10, {"code": "bad", "status": "error", "message": "x" * 2000,
                         "evidence": [{"path": "/p", "pointer": "/q", "value": list(range(5000))}]})
    report = {"kind": "k", "status": "error", "findings": findings}
    size = len(json.dumps(report).encode())
    assert size > RESULT_LIMIT
    summary = _summary(report, size)
    assert summary["status"] == "error" and summary["counts"]["error"] == 1
    assert summary["findings_shown"][0]["code"] == "bad" and summary["truncated_from_bytes"] == size
    assert len(json.dumps(summary).encode()) < 8 * 1024


def test_summary_preserves_unknown_repair_despite_successful_reproduction():
    report = {'kind': 'video_reference_recovery_reproduction', 'status': 'pass',
              'original_status': 'unknown', 'repair_verified': False,
              'reproduction_matches': True, 'verifier_matches': True,
              'assurance': 'internal_consistency_only', 'origin_authenticated': False,
              'origin_verified': False, 'verification_scope': 'retained_input_internal_consistency_only',
              'findings': [], 'scope': 'Internal replay only'}
    summary = _summary(report, RESULT_LIMIT + 1)
    for key in ('status', 'original_status', 'repair_verified', 'reproduction_matches', 'verifier_matches',
                'assurance', 'origin_authenticated', 'origin_verified', 'verification_scope'):
        assert summary[key] == report[key]
    assert summary['counts_scope'] == 'available_findings'


def test_summary_unicode_large_evidence_is_bounded_without_altering_counts():
    findings = [{'code': 'changed', 'status': 'different', 'message': '机器人🙂' * 1000,
                 'evidence': [{'path': '/' + '机器人🙂' * 1000, 'pointer': '/x', 'sha256': 'a' * 64}]}] * 30
    report = {'kind': 'comparison', 'status': 'different', 'findings': findings,
              'counts': {'different': 90}, 'total_findings': 90, 'findings_truncated': True,
              'candidate_status': 'different', 'candidate_evaluated': True}
    summary = _summary(report)
    assert len(json.dumps(summary).encode()) <= 8192
    assert len(json.dumps(summary, ensure_ascii=False).encode()) <= 8192
    assert summary['counts'] == report['counts'] and summary['status'] == 'different'
    assert summary['candidate_status'] == 'different' and summary['candidate_evaluated']
    assert summary['findings_total'] == 90
    assert summary['findings_omitted'] == 90 - len(summary['findings_shown'])


def test_oversized_mcp_replay_falls_back_without_certifying_unknown(tmp_path, monkeypatch):
    import policy_doctor.server as module

    report = {'kind': 'case_reproduction', 'status': 'pass', 'original_status': 'unknown',
              'repair_verified': False, 'reproduction_matches': True, 'verifier_matches': True,
              'padding': 'x' * RESULT_LIMIT}
    # The trajectory replay is a pre-existing default tool. This probes the generic
    # bound through its real FastMCP wrapper, rather than only the summary helper.
    monkeypatch.setattr(module, 'reproduce_case', lambda path: report)
    (tmp_path / 'case').mkdir()
    server = create_server([str(tmp_path)])
    async def run():
        return await server.call_tool('reproduce_trajectory_case', {'directory': str(tmp_path / 'case')})
    response = asyncio.run(run())
    body = response[1] if isinstance(response, tuple) else response
    assert body['status'] == 'pass' and body['original_status'] == 'unknown'
    assert body['repair_verified'] is False
    assert len(json.dumps(body).encode()) <= 8192


def test_setup_prints_configuration_and_writes_nothing(tmp_path):
    before = set(tmp_path.iterdir())
    out = subprocess.run([sys.executable, "-m", "policy_doctor", "setup", "--root", str(tmp_path), "--allow-hub"],
                         capture_output=True, text=True, cwd=tmp_path, env={"PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr
    assert "claude mcp add roboclue --" in out.stdout and "--allow-hub" in out.stdout
    config = json.loads(out.stdout.split("other clients (mcpServers JSON)\n", 1)[1].split("\n\n# Notes", 1)[0])
    assert config["mcpServers"]["roboclue"]["args"][:2] == ["--root", str(tmp_path.resolve())] or \
        config["mcpServers"]["roboclue"]["args"][-3:-1] == ["--root", str(tmp_path.resolve())]
    assert set(tmp_path.iterdir()) == before
    bad = subprocess.run([sys.executable, "-m", "policy_doctor", "setup", "--root", str(tmp_path / "missing")],
                         capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src")})
    assert bad.returncode == 2


def test_check_fail_on_gates(tmp_path):
    env = {"PYTHONPATH": str(ROOT / "src")}

    def run(*args):
        return subprocess.run([sys.executable, "-m", "policy_doctor", *args], capture_output=True, text=True, env=env)

    assert run("demo", str(tmp_path / "ex"), "--kind", "policy").returncode == 0
    cp, bad, good = (str(tmp_path / "ex" / x) for x in ("checkpoint", "dataset", "corrected_dataset"))
    assert run("check", cp, "--dataset", good).returncode == 3          # documented default: unknown deployment
    assert run("check", cp, "--dataset", good, "--fail-on", "error").returncode == 0
    assert run("check", cp, "--dataset", good, "--fail-on", "unknown").returncode == 1
    assert run("check", cp, "--dataset", bad, "--fail-on", "error").returncode == 1
