"""Bounded OpenCode retry/fallback policy for the v2 model adapter."""

from __future__ import annotations

from typing import Any

import pytest

import aa.conversation.model_adapter as adapter_mod
from aa.conversation.model_adapter import OpenCodeChatModel
from aa.opencode.client import SessionInfo
from aa.opencode.errors import OpenCodeProviderAccessError, OpenCodeRateLimitError, OpenCodeTransientError

PRIMARY = "opencode/space-bunny-free"
FALLBACK = "opencode/muse-spark-1.3-contributor-free"


class _ScriptedClient:
    def __init__(
        self,
        *,
        text_outcomes: list[object] | None = None,
        structured_outcomes: list[object] | None = None,
    ) -> None:
        self.text_outcomes = list(text_outcomes or [])
        self.structured_outcomes = list(structured_outcomes or [])
        self.models: list[str] = []
        self._sessions = 0

    async def create_session(self, title: str = "") -> SessionInfo:
        self._sessions += 1
        return SessionInfo(id=f"ses_{self._sessions}", title=title)

    async def delete_session(self, session_id: str) -> bool:
        return True

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        **_: Any,
    ) -> str:
        self.models.append(model)
        outcome = self.text_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)

    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        **_: Any,
    ) -> dict[str, object]:
        self.models.append(model)
        outcome = self.structured_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, dict)
        return outcome


async def test_text_429_escalates_immediately_for_fresh_runner_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(text_outcomes=[OpenCodeRateLimitError("http=429")])
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(adapter_mod.asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    with pytest.raises(OpenCodeRateLimitError):
        await model._ainvoke_text("hello")
    assert client.models == [PRIMARY]
    assert sleeps == []


async def test_text_403_retries_with_exponential_backoff_then_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        text_outcomes=[
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            "fallback-ok",
        ]
    )
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(adapter_mod.asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    assert await model._ainvoke_text("hello") == "fallback-ok"
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, PRIMARY, FALLBACK]
    assert sleeps == [5.0, 10.0, 20.0]


async def test_structured_429_escalates_immediately_for_fresh_runner_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(structured_outcomes=[OpenCodeRateLimitError("http=429")])
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(adapter_mod.asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    with pytest.raises(OpenCodeRateLimitError):
        await model.ainvoke_structured(
            "hello",
            system="planner",
            schema={"type": "object"},
        )
    assert client.models == [PRIMARY]
    assert sleeps == []
