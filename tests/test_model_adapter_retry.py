"""Bounded OpenCode retry/fallback policy for the v2 model adapter."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from aa.conversation.model_adapter import (
    PRIMARY_ACCESS_CIRCUIT_TTL_S,
    OpenCodeChatModel,
    clear_primary_circuit,
)
from aa.opencode.client import SessionInfo
from aa.opencode.errors import (
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
    OpenCodeTransientError,
)

PRIMARY = "opencode/space-bunny-free"
FALLBACK = "opencode/muse-spark-1.3-contributor-free"


@pytest.fixture(autouse=True)
def _clear_shared_primary_circuit() -> Any:
    """Isolate the process-wide primary circuit between tests."""
    clear_primary_circuit()
    yield
    clear_primary_circuit()


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

    monkeypatch.setattr(asyncio, "sleep", _sleep)
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
    # Live SLO guard: a persistent 403 for the pinned primary must fail over
    # fast (single bounded >=5s retry, then fallback) instead of burning 35s
    # of sleep per provider call (Gate C run 37498373507: p50 50s/p95 86s).
    # Gate C repair (run 37608207212): custom-selector 403 first retries the
    # same strong primary with the selector omitted before the weak
    # fallback, so the scripted primary exhausts custom (2x) + omitted (1x)
    # before fallback.
    client = _ScriptedClient(
        text_outcomes=[
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            "fallback-ok",
        ]
    )
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    assert await model._ainvoke_text("hello") == "fallback-ok"
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK]
    assert sleeps == [5.0]
    assert sleeps[0] >= 5.0


async def test_structured_429_escalates_immediately_for_fresh_runner_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(structured_outcomes=[OpenCodeRateLimitError("http=429")])
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
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


async def test_persistent_403_skips_redundant_primary_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent primary 403 (Gate C 37504648482) must fast-failover.

    The first call performs the required >=5s retry then fallback. Later
    calls in the same process go directly to fallback without burning
    another 5s sleep plus slow primary attempts per call (15s overhead
    per ordinary turn). Gate C repair (37608207212) tries the omitted
    primary before the weak fallback on both slow and fast paths, so each
    exhausted primary costs custom (2x) + omitted (1x) before fallback on
    the slow path and omitted (1x) before fallback on the fast path.
    """
    client = _ScriptedClient(
        text_outcomes=[
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            "fallback-ok-1",
            OpenCodeProviderAccessError("http=403"),
            "fallback-ok-2",
        ]
    )
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    assert await model._ainvoke_text("hello") == "fallback-ok-1"
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK]
    assert sleeps == [5.0]

    assert await model._ainvoke_text("hello again") == "fallback-ok-2"
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK, PRIMARY, FALLBACK]
    assert sleeps == [5.0]


async def test_primary_recovery_closes_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovered primary must be retried after the circuit TTL."""
    client = _ScriptedClient(
        text_outcomes=[
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            "fallback-ok",
            "primary-ok",
        ]
    )
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    assert await model._ainvoke_text("hello") == "fallback-ok"
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK]
    # Force circuit expiry without waiting for the TTL. Setting the stamp to
    # 0.0 is not sufficient: time.monotonic() on a freshly booted runner can
    # be below the TTL, leaving the circuit open. Expire relative to now.
    # The circuit is process-wide per primary model, so expire the shared
    # entry as well as the per-instance stamp.
    assert model._primary_access_rejected_at is not None
    from aa.conversation import model_adapter as _ma

    _expired = time.monotonic() - PRIMARY_ACCESS_CIRCUIT_TTL_S - 1.0
    model._primary_access_rejected_at = _expired
    _ma._PRIMARY_CIRCUIT[_ma._circuit_key(PRIMARY)] = _expired
    assert await model._ainvoke_text("hello again") == "primary-ok"
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK, PRIMARY]
    assert model._primary_access_rejected_at is None


async def test_transient_fallback_does_not_open_primary_403_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        text_outcomes=[
            OpenCodeTransientError("temporary"),
            OpenCodeTransientError("temporary"),
            OpenCodeTransientError("temporary"),
            "fallback-ok",
            "primary-ok",
        ]
    )
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    assert await model._ainvoke_text("hello") == "fallback-ok"
    assert model._primary_access_rejected_at is None
    assert await model._ainvoke_text("hello again") == "primary-ok"
    assert client.models[-1] == PRIMARY


async def test_fallback_403_clears_primary_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        text_outcomes=[
            OpenCodeProviderAccessError("primary-403"),
            OpenCodeProviderAccessError("primary-403"),
            OpenCodeProviderAccessError("omitted-403"),
            OpenCodeProviderAccessError("fallback-403"),
            "primary-recovered",
        ]
    )
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    with pytest.raises(OpenCodeProviderAccessError):
        await model._ainvoke_text("hello")
    assert model._primary_access_rejected_at is None
    assert await model._ainvoke_text("hello again") == "primary-recovered"
    assert client.models[-1] == PRIMARY


async def test_structured_fallback_403_clears_primary_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        structured_outcomes=[
            OpenCodeProviderAccessError("primary-403"),
            OpenCodeProviderAccessError("primary-403"),
            OpenCodeProviderAccessError("omitted-403"),
            OpenCodeProviderAccessError("fallback-403"),
            {"ok": True},
        ]
    )
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    with pytest.raises(OpenCodeProviderAccessError):
        await model.ainvoke_structured(
            "hello",
            system="planner",
            schema={"type": "object"},
        )
    assert model._primary_access_rejected_at is None
    assert await model.ainvoke_structured(
        "hello again",
        system="planner",
        schema={"type": "object"},
    ) == {"ok": True}
    assert client.models[-1] == PRIMARY


async def test_cross_agent_shares_primary_403_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate C regression: verifier must fast-fallback after planner 403.

    Run 37519360307 showed p50 17s/p95 36s/max 45s with verifier never
    served and 14 generic clarifications: each agent burned its own primary
    retry per turn. A planner rejection must open the process-wide circuit
    so answer/verifier bound via with_agent go directly to fallback.
    Gate C repair (37608207212) tries omitted primary before weak fallback
    on both paths, so exhausted primaries cost an extra omitted PRIMARY.
    """
    client = _ScriptedClient(
        text_outcomes=[
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            "planner-fallback-ok",
            OpenCodeProviderAccessError("http=403"),
            "answer-fallback-ok",
        ]
    )
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    planner = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    assert await planner._ainvoke_text("planner hello") == "planner-fallback-ok"
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK]
    assert sleeps == [5.0]

    answer = planner.with_agent("aa-v2")
    assert await answer._ainvoke_text("answer hello") == "answer-fallback-ok"
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK, PRIMARY, FALLBACK]
    assert sleeps == [5.0]


async def test_with_agent_created_after_rejection_fast_fallbacks_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured verifier bound after planner 403 must skip primary retry."""
    client = _ScriptedClient(
        structured_outcomes=[
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            {"queries": ["q0"] * 12},
            OpenCodeProviderAccessError("http=403"),
            {"units": [], "all_required_supported": True},
        ]
    )
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    planner = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    assert await planner.ainvoke_structured("planner", system="p", schema={"type": "object"}) == {
        "queries": ["q0"] * 12
    }
    assert sleeps == [5.0]

    verifier = planner.with_agent("aa-verifier-v2")
    assert await verifier.ainvoke_structured("verifier", system="v", schema={"type": "object"}) == {
        "units": [],
        "all_required_supported": True,
    }
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK, PRIMARY, FALLBACK]
    assert sleeps == [5.0]
