"""P0 kodmial/aa#204 regression: repair Gate C live-path-failed.

Live evidence on exact main 8c57bba (run 37608207212) showed the
live-telegram-evidence lane failing with the concrete component signature:

- live-answer-no-generic-collapse (13 generic clarifications of 14 turns),
- live-answer-diversity (distinct replies below the diversity floor),
- live-text-max-over-budget (max 30.7s over the 30s hard budget),

with stage outcomes verifier ``unavailable`` 13/14 and answer
``clarification`` 13/14 while planner/answer/summarizer served through the
weak Space Bunny fallback and the decoupled verifier (omitted selector)
served Muse Spark. The verifier probe already isolates this as an
agent-selector rejection (custom selector 403, omitted serves).

The repair retries the same strong primary with the transport selector
omitted before falling back to the weak model, on both slow and fast
(circuit-open) paths and for both text and structured calls. This keeps
grounding strong without any exact-question special case and without
weakening Product Contract #110. Provider 429 always propagates for
runner retire/restart and never triggers the omitted path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aa.conversation.model_adapter import OpenCodeChatModel, clear_primary_circuit
from aa.conversation.verifier import clear_verifier_capability_cache
from aa.opencode.client import SessionInfo
from aa.opencode.errors import (
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


class _SelectorClient:
    """Scripted client distinguishing custom vs omitted selector.

    Custom-selector primary calls (agent non-empty) fail with 403 to
    reproduce the live agent-selector rejection; omitted-selector primary
    calls (agent empty, audit preserved) serve the strong primary; weak
    fallback calls serve fallback. Only privacy-safe identities are
    recorded (wire agent, audit agent, model).
    """

    def __init__(self) -> None:
        self.wire_agents: list[str] = []
        self.audit_agents: list[str] = []
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
        audit_agent: str = "",
        **_: Any,
    ) -> str:
        self.wire_agents.append(agent)
        self.audit_agents.append(audit_agent)
        self.models.append(model)
        if model == PRIMARY and agent:
            raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")
        if model == PRIMARY and not agent:
            return "omitted-primary-ok"
        return "fallback-ok"

    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        agent: str = "",
        audit_agent: str = "",
        **_: Any,
    ) -> dict[str, object]:
        self.wire_agents.append(agent)
        self.audit_agents.append(audit_agent)
        self.models.append(model)
        if model == PRIMARY and agent:
            raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")
        if model == PRIMARY and not agent:
            return {"queries": ["strong query"] * 12}
        return {"queries": ["fallback query"] * 12}


async def test_text_custom_403_serves_omitted_primary_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom 403 must serve the same strong primary omitted, not weak fallback."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client = _SelectorClient()
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    assert await model._ainvoke_text("hello", system="answer-system") == "omitted-primary-ok"
    # Custom primary tried (initial + >=5s retry), then omitted primary serves.
    assert client.models == [PRIMARY, PRIMARY, PRIMARY]
    assert client.wire_agents == ["aa-v2", "aa-v2", ""]
    # Logical audit stays aa-v2 while the wire omits the selector.
    assert client.audit_agents == ["", "", "aa-v2"]
    assert sleeps == [5.0]


async def test_structured_custom_403_serves_omitted_primary_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner structured custom 403 must serve omitted primary, not fallback."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client = _SelectorClient()
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    result = await model.ainvoke_structured("hello", system="planner", schema={"type": "object"})
    assert result == {"queries": ["strong query"] * 12}
    assert client.models == [PRIMARY, PRIMARY, PRIMARY]
    assert client.wire_agents == ["aa-planner-v2", "aa-planner-v2", ""]
    assert client.audit_agents == ["", "", "aa-planner-v2"]
    assert sleeps == [5.0]


async def test_fast_path_omitted_primary_avoids_weak_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Circuit-open turns must stay strong via omitted primary (no 5s retry)."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client = _SelectorClient()
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    assert await model._ainvoke_text("first", system="s") == "omitted-primary-ok"
    assert sleeps == [5.0]
    # Second turn: circuit open, omitted primary serves directly, no sleep.
    assert await model._ainvoke_text("second", system="s") == "omitted-primary-ok"
    assert sleeps == [5.0]
    assert client.models[-1] == PRIMARY
    assert client.wire_agents[-1] == ""
    assert client.audit_agents[-1] == "aa-v2"


async def test_omitted_429_propagates_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 429 on the omitted path must retire the runner, never fallback."""

    class _Limited(_SelectorClient):
        async def send_message(self, session_id: str, text: str, **kwargs: Any) -> str:
            self.wire_agents.append(str(kwargs.get("agent", "")))
            self.audit_agents.append(str(kwargs.get("audit_agent", "")))
            self.models.append(str(kwargs.get("model", "")))
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        _Limited(),  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    with pytest.raises(OpenCodeRateLimitError):
        await model._ainvoke_text("hello", system="s")
    assert sleeps == []
