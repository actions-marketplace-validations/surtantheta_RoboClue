"""Regression checks for report failures found by simulated beta operators."""

import copy

import pytest
from test_video_recovery import PublicSource, create
from test_windows import make_dataset

from policy_doctor import video_recovery as recovery
from policy_doctor.cli import main
from policy_doctor.reports import markdown_report
from policy_doctor.windows import audit_episode_windows


def test_native_video_recovery_creator_replay_and_later_case_render(monkeypatch, tmp_path, capsys):
    source = PublicSource()
    first, created, _ = create(monkeypatch, tmp_path, source, name="first")
    second, _, _ = create(monkeypatch, tmp_path, source, episode=1, name="second")
    monkeypatch.setattr(recovery, "_opener", lambda: pytest.fail("Unexpected network"))
    reports = [created, recovery.reproduce_video_recovery(first),
               recovery.check_video_recovery_candidate(first, second)]
    for result in reports:
        before = copy.deepcopy(result)
        rendered = markdown_report(result)
        assert result == before
        assert "Original interval status: **error**" in rendered
        assert "Repair verified: **True**" in rendered
        assert "Origin verified: **False**" in rendered
        assert "physical execution" in rendered.lower()
        assert "## Findings" not in rendered
    assert "| Old | New |" in markdown_report(created)
    assert "Report reproduced: True" in markdown_report(reports[1])
    assert "Candidate episode: 1" in markdown_report(reports[2])
    assert main(['reproduce-case', str(first), '--markdown']) == 0
    assert 'Report reproduced: True' in capsys.readouterr().out
    assert main(['check-case', str(first), str(second), '--markdown']) == 0
    assert 'Candidate episode: 1' in capsys.readouterr().out


def test_unresolved_native_video_case_does_not_claim_verified_repair(monkeypatch, tmp_path):
    _, result, _ = create(monkeypatch, tmp_path, PublicSource(healthy=True))
    rendered = markdown_report(result)
    assert "Report status: **unknown**" in rendered
    assert "Original interval status: **pass**" in rendered
    assert "Repair verified: **False**" in rendered
    assert "in bounds" in rendered


def test_declared_totals_report_retains_both_values_and_native_explanation(tmp_path):
    root = make_dataset(tmp_path, mutate=lambda info, rows, meta: info.update(
        total_frames=0, total_episodes=0))
    result = audit_episode_windows(root)
    rendered = markdown_report(result)
    assert "declared 0; actual 60" in rendered
    assert "declared 0; actual 2" in rendered
    assert "actual None" not in rendered and "episode None" not in rendered
    assert "meta/info.json declares" in rendered
    assert "corrected window" not in rendered.lower()
    assert "## Selected window" not in rendered


def test_span_contradiction_keeps_distinct_declared_and_actual_window(tmp_path):
    root = make_dataset(tmp_path, bad=True)
    result = audit_episode_windows(root, episode_index=0, anchor_index=29, delta_indices=[0, 1])
    rendered = markdown_report(result)
    assert "episode 0" in rendered and "declared 33; actual 30" in rendered
    assert "## Selected window" in rendered
    assert "The corrected window is a numerical what-if" in rendered
    assert "action\\_is\\_pad" in rendered


def test_short_episode_warning_renders_message_without_invented_values(tmp_path):
    def one_frame(info, rows, meta):
        rows[:] = rows[:1]
        meta[:] = [dict(meta[0], length=1, dataset_to_index=1)]
        info.update(total_frames=1, total_episodes=1)

    result = audit_episode_windows(make_dataset(tmp_path, mutate=one_frame))
    warning = next(f for f in result["findings"] if f["code"] == "episode.too_short")
    rendered = markdown_report(result)
    assert warning["message"].split(":", 1)[0] in rendered
    assert r"\[0\]" in rendered
    assert "declared None" not in rendered and "actual None" not in rendered
    assert "corrected window" not in rendered.lower()


def test_markdown_quotes_remain_readable_but_untrusted_html_and_links_are_data(tmp_path):
    root = make_dataset(tmp_path)
    result = audit_episode_windows(root)
    result["scope"] = 'Untrusted <script>alert("x")</script> & [click](javascript:alert(1)) | "quoted"'
    result["coverage"]["quoted"] = 'robot\'s "camera"'
    rendered = markdown_report(result)
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered
    assert '&amp;' in rendered and '\\[click\\]' in rendered and '\\|' in rendered
    assert '"quoted"' in rendered and "robot's" in rendered
    assert '&quot;' not in rendered and '&#x27;' not in rendered
