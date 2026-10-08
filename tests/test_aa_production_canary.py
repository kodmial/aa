"""Unit contracts for the gated AA production canary runner (issue #82)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_aa_production_canary import (  # noqa: E402
    assert_evidence_privacy_safe,
    classify_failure,
    evaluate_activation,
)


def test_activation_requires_all_three_gates() -> None:
    assert (
        evaluate_activation(
            issue6_closed=True, qual_pass_for_sha=True, contract_status="ready"
        ).active
        is True
    )
    assert (
        evaluate_activation(
            issue6_closed=False, qual_pass_for_sha=True, contract_status="ready"
        ).active
        is False
    )
    assert (
        evaluate_activation(
            issue6_closed=True, qual_pass_for_sha=False, contract_status="ready"
        ).active
        is False
    )
    assert (
        evaluate_activation(
            issue6_closed=True, qual_pass_for_sha=True, contract_status="not-activated"
        ).active
        is False
    )


def test_failure_taxonomy_covers_required_classes() -> None:
    assert classify_failure("live-answer-failed") == "product-regression"
    assert classify_failure("OPENCODE_429_RESTART_REQUIRED") == "provider-transient"
    assert classify_failure("runner-network-timeout") == "github-infrastructure"
    assert classify_failure("main-changed-during-run") == "stale-main"
    assert classify_failure("not-activated-missing-qual") == "not-activated"


def test_evidence_rejects_privacy_sensitive_keys() -> None:
    try:
        assert_evidence_privacy_safe({"transcript": "secret words"})
    except ValueError:
        pass
    else:
        raise AssertionError("transcript evidence must be rejected")
    assert_evidence_privacy_safe({"failed_checks": ["meta-turn"], "run_id": "7"})


def test_canary_skipped_before_activation() -> None:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=True,
    ).stdout.strip()
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_aa_production_canary.py",
            "--main-sha",
            sha,
            "--run-id",
            "unit-skipped",
            "--out-dir",
            "/tmp/aa-canary-unit-skipped",
            "--issue6-closed",
            "false",
            "--qual-pass",
            "false",
            "--current-main-sha",
            sha,
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert proc.returncode == 2
    payload = json.loads(Path("/tmp/aa-canary-unit-skipped/result.json").read_text())
    assert payload["result"] == "SKIPPED"


async def test_compact_canary_passes_hermetically(monkeypatch: object) -> None:
    import urllib.request
    from typing import Any

    from scripts.run_aa_production_canary import _run_compact_checks

    # Fail-closed telegram readiness requires a token to prove Bot API
    # authentication: stub the token plus the getMe transport so the
    # hermetic run proves the live auth path without network I/O.
    _patch: Any = monkeypatch
    _patch.setenv("TELEGRAM_BOT_TOKEN", "123456:canary-test-token")
    real_urlopen: Any = urllib.request.urlopen

    class _FakeGetMe:
        def __enter__(self) -> _FakeGetMe:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"id": 1, "is_bot": True}}).encode("utf-8")

    def _fake_urlopen(request: Any, timeout: Any = None) -> Any:
        try:
            url = str(getattr(request, "full_url", request))
        except Exception:
            url = str(request)
        if "api.telegram.org" in url:
            return _FakeGetMe()
        return real_urlopen(request, timeout=timeout)

    _patch.setattr(urllib.request, "urlopen", _fake_urlopen)
    checks, startup_ms = await _run_compact_checks()
    assert startup_ms >= 0
    names = {check.name for check in checks}
    assert {
        "corpus-prerequisites",
        "retrieval-contract",
        "production-startup",
        "opencode-readiness",
        "telegram-readiness",
        "meta-turn",
        "substantive-turn",
        "followup-continuity",
        "no-legacy-routing",
        "safety-fixture",
        "session-isolation",
        "typing-heartbeat",
        "voice-fixture",
        "clean-shutdown",
    } <= names
    assert all(check.status == "PASS" for check in checks), [
        check.to_dict() for check in checks if check.status != "PASS"
    ]


async def test_telegram_readiness_fails_closed_without_token(monkeypatch: object) -> None:
    from typing import Any

    from scripts.run_aa_production_canary import _run_compact_checks

    _patch: Any = monkeypatch
    _patch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    checks, _ = await _run_compact_checks()
    by_name = {check.name: check for check in checks}
    assert by_name["telegram-readiness"].status == "FAIL"
    assert by_name["telegram-readiness"].detail == "telegram-token-missing"
