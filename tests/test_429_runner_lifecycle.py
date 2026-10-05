"""Global AA OpenCode 429 runner-lifecycle contract."""

from __future__ import annotations

from pathlib import Path

import pytest

import aa.__main__ as main_mod
from aa.conversation.orchestrator import send_with_fallback
from aa.opencode.errors import (
    OpenCodeRateLimitError,
    classify_http_status,
    classify_provider_error,
)

ROOT = Path(__file__).resolve().parents[1]


def test_http_and_provider_429_use_dedicated_rate_limit_error() -> None:
    assert isinstance(classify_http_status(429), OpenCodeRateLimitError)
    assert isinstance(
        classify_provider_error(
            {"name": "APIError", "data": {"statusCode": 429, "isRetryable": True}}
        ),
        OpenCodeRateLimitError,
    )


@pytest.mark.asyncio
async def test_synthesis_429_never_retries_or_uses_fallback() -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    async def send(
        session_id: str,
        prompt: str,
        *,
        agent: str = "",
        model: str = "",
        **_: object,
    ) -> str:
        calls.append(model)
        raise OpenCodeRateLimitError("http=429")

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    with pytest.raises(OpenCodeRateLimitError):
        await send_with_fallback(
            send,
            "ses_1",
            "prompt",
            agent="aa",
            primary_model="opencode/space-bunny-free",
            fallback_model="opencode/muse-spark-1.3-contributor-free",
            sleep=sleep,
        )

    assert calls == ["opencode/space-bunny-free"]
    assert sleeps == []


def test_cli_maps_rate_limit_to_runner_restart_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Settings:
        log_level = "INFO"
        has_bot_token = True

        def validate(self, *, require_bot_token: bool = False) -> None:
            _ = require_bot_token

    async def _boom(_settings: object) -> int:
        raise OpenCodeRateLimitError("http=429")

    monkeypatch.setattr(main_mod.Settings, "from_env", lambda: _Settings())
    monkeypatch.setattr(main_mod, "configure_logging", lambda _level: None)
    monkeypatch.setattr(main_mod, "_run_worker", _boom)

    assert main_mod.main([]) == 75


def test_runtime_workflow_emits_429_recovery_artifact() -> None:
    workflow = (ROOT / ".github" / "workflows" / "aa-runtime.yml").read_text(encoding="utf-8")
    assert 'if [ "$status" -eq 75 ]' in workflow
    assert "OPENCODE_429_RESTART_REQUIRED" in workflow
    assert "aa-runtime-429-recovery" in workflow
    assert "opencode/space-bunny-free" in workflow


def test_global_429_recovery_watches_runtime_and_qualification() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "aa-real-book-429-recovery.yml"
    ).read_text(encoding="utf-8")
    assert '"AA real-book retrieval qualification"' in workflow
    assert '"AA bot runtime"' in workflow
    assert "aa-runtime-429-recovery" in workflow
    assert "run_attempt < 4" in workflow
    assert "reRunWorkflow" in workflow
