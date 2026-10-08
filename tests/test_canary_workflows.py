"""Regression contracts for the gated AA production canary (issue #82)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_canary_is_single_gated_workflow_every_four_hours() -> None:
    assert not (WORKFLOWS / "aa-canary-monitor.yml").exists()
    text = _read("aa-canary.yml")
    assert 'cron: "17 */4 * * *"' in text
    assert "workflow_dispatch:" in text
    assert "github.repository_owner" in text
    assert "group: aa-production-canary" in text
    assert "cancel-in-progress: false" in text


def test_canary_activation_enforces_blocker_before_any_runtime() -> None:
    text = _read("aa-canary.yml")
    assert "issues/6" in text or "issue_number: 6" in text
    assert "continuum-qualification-result issue=7" in text
    assert "verify_product_contract_qualification" in text
    assert "SKIPPED / not activated" in text
    assert "needs.activation.outputs.active" in text
    assert "run_aa_production_canary.py" in text


def test_canary_scope_uses_real_production_boundary() -> None:
    text = _read("aa-canary.yml")
    assert "bash scripts/verify.sh" in text
    assert "restore_canonical.py" in text
    assert "build_retrieval_index.py" in text
    assert "TELEGRAM_BOT_TOKEN" in text
    assert "AA_BOOK_AGE_IDENTITY" in text


def test_canary_stale_and_transient_never_create_repair_issues() -> None:
    text = _read("aa-canary.yml")
    assert "STALE" in text
    assert "requalification pending" in text
    assert "provider/infrastructure transient" in text
    assert "no product repair issue is created" in text


def test_canary_repair_is_single_p0_with_recovery() -> None:
    text = _read("aa-canary.yml")
    assert "aa-production-canary-repair:v1" in text
    assert "Canary fingerprint:" in text
    assert "priority:p0" in text
    assert "continuum-issue-scheduler.yml" in text
    assert "Close matching canary repair issues on recovery" in text
    assert "No user text, secrets, corpus text, audio, transcripts" in text


def test_canary_runner_exists_with_gated_entrypoint() -> None:
    runner = ROOT / "scripts" / "run_aa_production_canary.py"
    assert runner.is_file()
    text = runner.read_text(encoding="utf-8")
    assert "aa-production-canary/1" in text
    assert "evaluate_activation" in text
    assert "classify_failure" in text
    assert "not-activated" in text
    assert "stale-main" in text
