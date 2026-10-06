"""Regression contracts for the recurring AA canary workflows."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_canary_runs_every_four_hours_only_after_final_qualification() -> None:
    text = _read("aa-canary.yml")
    assert 'cron: "17 */4 * * *"' in text
    assert "workflow_dispatch:" in text
    # Final qualification gate is capability #6 plus exact-main
    # qualification #7 (commit ff43249). Legacy #80 is superseded and must
    # never activate the canary, so no activation may reference it.
    assert "issue_number: 6" in text
    assert "issue_number: 7" in text
    assert "issue_number: 80" not in text
    assert "capability.state === 'closed'" in text
    assert "qualification.state === 'closed'" in text
    assert "state_reason === 'completed'" in text
    assert "needs: activation" in text
    assert "needs.activation.outputs.active == 'true'" in text
    assert "canary stays inactive" in text
    assert "bash scripts/verify.sh" in text
    assert "test_spawned_server_shuts_down_gracefully" in text
    assert "test_attach_mode_never_kills_foreign_server" in text
    assert "/getMe" in text
    assert "TELEGRAM_BOT_TOKEN" in text
    assert "release" not in text.lower().replace("product release: not performed", "")


def test_canary_monitor_deduplicates_one_p0_repair_issue() -> None:
    text = _read("aa-canary-monitor.yml")
    assert 'workflows: ["AA canary"]' in text
    assert "aa-canary-repair:v1" in text
    assert "priority:p0" in text
    assert "listForRepo" in text
    assert "issues.update" in text
    assert "issues.create" in text
    assert "duplicate canary failures must reuse it" in text
