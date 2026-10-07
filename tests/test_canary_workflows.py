"""Regression contracts for the recurring AA canary workflows."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_canary_runs_every_four_hours_during_convergence() -> None:
    text = _read("aa-canary.yml")
    assert 'cron: "17 */4 * * *"' in text
    assert "workflow_dispatch:" in text
    assert "group: aa-self-proving-qualification" in text
    assert "Activate bounded current-main canary" in text
    assert "active=true" in text
    assert "needs: activation" in text
    assert "needs.activation.outputs.active == 'true'" in text
    assert "Gate C smoke" in text
    assert "Gate D readiness" in text
    assert "Gate E latency guard" in text
    assert "Re-enter authoritative convergence after canary failure" in text
    assert "aa-self-proving-qualification.yml" in text
    assert "bash scripts/verify.sh" in text
    assert "test_spawned_server_shuts_down_gracefully" in text
    assert "test_attach_mode_never_kills_foreign_server" in text
    assert "/getMe" in text
    assert "TELEGRAM_BOT_TOKEN" in text
    assert "release" not in text.lower().replace("product release: not performed", "")


def test_canary_monitor_routes_failure_to_authoritative_self_proving_only() -> None:
    text = _read("aa-canary-monitor.yml")
    assert 'workflows: ["AA canary"]' in text
    assert "Re-enter authoritative self-proving convergence" in text
    assert "actions: write" in text
    assert "aa-self-proving-qualification.yml" in text
    assert "listWorkflowRuns" in text
    assert "createWorkflowDispatch" in text
    assert "issues.create" not in text
    assert "priority:p0" not in text
    assert "No generic canary P0 is created" in text
