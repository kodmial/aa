"""P0 kodmial/aa#206 regression: omitted retry must stay fallback-safe.

Live evidence on exact main e330f2b (run 37612325642) showed the
live-telegram-evidence lane failing with the concrete component signature:

- live-answer-diversity (distinct replies below the diversity floor),
- live-actual-served-model-identity (only aa-summarizer-v2 served),
- live-verifier-muse-access-failure (no served aa-verifier-v2),
- live-planner-retrieval-answer-verifier-telemetry (0 stage snapshots),

with 14 fast natural-fallback turns (p50 0.35s), clarification 0, and no
planner/answer/verifier audit. The e330f omitted-primary repair only let
access/transient/timeout omitted failures fall through to the configured
weak fallback; any deterministic omitted failure (structured-output-missing,
served-model-mismatch, not-ready, session-not-found) propagated and stranded
the fallback, collapsing the graph to retry replies with no telemetry.

The repair keeps the strong omitted attempt but lets every omitted failure
except 429 fall through to the weak fallback, on both slow and fast
(circuit-open) paths and for both text and structured calls. Provider 429
always propagates for runner retire/restart and never triggers fallback.
Product Contract #110 stays immutable; no exact-question special cases.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aa.conversation.model_adapter import OpenCodeChatModel, clear_primary_circuit
from aa.conversation.verifier import clear_verifier_capability_cache
from aa.opencode.client import SessionInfo
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeNotReadyError,
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
)

PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"


@pytest.fixture(autouse=True)
def _clear_state() -> Any:
    clear_primary_circuit()
    clear_verifier_capability_cache()
    yield
    clear_primary_circuit()
    clear_verifier_capability_cache()


class _FallbackSafeClient:
    """Distinguishes custom vs omitted selector with scripted outcomes.

    Custom-selector primary calls fail with 403; omitted-selector primary
    calls fail with the injected omitted failure; weak fallback calls serve.
    Records only privacy-safe identities (wire agent, model).
    """

    def __init__(self, *, omitted_failure: BaseException) -> None:
        self._omitted_failure = omitted_failure
        self.wire_agents: list[str] = []
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
        agent: str = "",
        **_: Any,
    ) -> str:
        self.wire_agents.append(agent)
        self.models.append(model)
        if model == PRIMARY and agent:
            raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")
        if model == PRIMARY and not agent:
            raise self._omitted_failure
        return "fallback-ok"

    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        agent: str = "",
        **_: Any,
    ) -> dict[str, object]:
        self.wire_agents.append(agent)
        self.models.append(model)
        if model == PRIMARY and agent:
            raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")
        if model == PRIMARY and not agent:
            raise self._omitted_failure
        return {"queries": ["fallback query"] * 12}


async def test_text_omitted_deterministic_still_serves_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow path: custom 403 + omitted deterministic must serve fallback."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client = _FallbackSafeClient(
        omitted_failure=OpenCodeDeterministicError("opencode structured output missing"),
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    assert await model._ainvoke_text("hello", system="s") == "fallback-ok"
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK]
    assert client.wire_agents == ["aa-v2", "aa-v2", "", "aa-v2"]
    assert sleeps == [5.0]


async def test_structured_omitted_deterministic_still_serves_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner structured: omitted deterministic must not strand fallback."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client = _FallbackSafeClient(
        omitted_failure=OpenCodeDeterministicError("opencode structured output missing"),
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    result = await model.ainvoke_structured("hello", system="p", schema={"type": "object"})
    assert result == {"queries": ["fallback query"] * 12}
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK]
    assert sleeps == [5.0]


async def test_fast_path_omitted_not_ready_still_serves_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Circuit-open path: omitted not-ready must fall through to fallback."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client = _FallbackSafeClient(
        omitted_failure=OpenCodeNotReadyError("opencode runtime has not been started"),
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    model._record_primary_rejection()
    assert await model._ainvoke_text("hello", system="s") == "fallback-ok"
    assert sleeps == []
    assert client.models[-2:] == [PRIMARY, FALLBACK]
    assert client.wire_agents[-2:] == ["", "aa-v2"]


async def test_omitted_429_still_propagates_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 429 on the omitted path must retire the runner, never fallback."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client = _FallbackSafeClient(
        omitted_failure=OpenCodeRateLimitError("opencode request rate-limited: http=429"),
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    with pytest.raises(OpenCodeRateLimitError):
        await model._ainvoke_text("hello", system="s")
    assert sleeps == [5.0]
    assert FALLBACK not in client.models
