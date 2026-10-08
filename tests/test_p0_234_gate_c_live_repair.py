"""Gate C live repair for kodmial/aa#234.

Live run 37725833818 on exact main 1563b77 failed with
``C:live-answer-no-generic-collapse``: planner/answer served only the weak
fallback while the verifier (Muse-only) was never served, with 6 fast
verifier-unavailable turns (verifier p50 0.3ms, response_units_total=0)
collapsing to 5 generic clarifications. Planner/retrieval stayed healthy
(``planner ok``/``evidence-ready`` 10/10), so the defect is the
fallback/circuit boundary, not retrieval or grounding strictness.

Omitted-wire models (empty transport selector, the production planner/
answer/summarizer/verifier wiring) went directly to the weak fallback on
every call while the shared primary circuit stayed open for the full TTL,
never re-probing the pinned primary even after it recovered. Custom-wire
models already re-probe the omitted primary while the circuit is open; the
omitted-wire path must do the same single no-sleep re-probe before falling
back so one transient 403 cannot pin the whole run to weak drafts.
"""

from __future__ import annotations

import asyncio

import pytest

from aa.conversation.model_adapter import OpenCodeChatModel, clear_primary_circuit
from aa.opencode.client import SessionInfo
from aa.opencode.errors import OpenCodeProviderAccessError


@pytest.fixture(autouse=True)
def _clear_shared_primary_circuit() -> object:
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

    async def create_session(self, title: str = "") -> SessionInfo:
        return SessionInfo(id="ses", title=title)

    async def delete_session(self, session_id: str) -> bool:
        return True

    async def send_message(self, session_id: str, text: str, **kwargs: object) -> str:
        model = str(kwargs.get("model", ""))
        self.models.append(model)
        outcome = self.text_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)

    async def send_structured_message(
        self, session_id: str, text: str, **kwargs: object
    ) -> dict[str, object]:
        model = str(kwargs.get("model", ""))
        self.models.append(model)
        outcome = self.structured_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, dict)
        return outcome


PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"


async def test_omitted_wire_text_reprobes_primary_while_circuit_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        text_outcomes=[
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            "fallback-ok",
            "primary-recovered",
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
        transport_agent="",
    )

    assert await model._ainvoke_text("hello") == "fallback-ok"
    assert client.models == [PRIMARY, PRIMARY, FALLBACK]
    assert sleeps == [5.0]

    # Circuit is open, but the already-omitted primary must be re-probed
    # once with no sleep before falling back, so a recovered primary
    # restores strong serves instead of pinning the run to weak fallback.
    assert await model._ainvoke_text("hello again") == "primary-recovered"
    assert client.models == [PRIMARY, PRIMARY, FALLBACK, PRIMARY]
    assert sleeps == [5.0]


async def test_omitted_wire_text_falls_back_when_reprobe_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        text_outcomes=[
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
        transport_agent="",
    )

    assert await model._ainvoke_text("hello") == "fallback-ok-1"
    assert await model._ainvoke_text("hello again") == "fallback-ok-2"
    assert client.models == [PRIMARY, PRIMARY, FALLBACK, PRIMARY, FALLBACK]
    assert sleeps == [5.0]


async def test_omitted_wire_structured_reprobes_primary_while_circuit_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedClient(
        structured_outcomes=[
            OpenCodeProviderAccessError("http=403"),
            OpenCodeProviderAccessError("http=403"),
            {"fallback": True},
            {"primary": True},
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
        transport_agent="",
    )

    first = await model.ainvoke_structured("hello", system="p", schema={"type": "object"})
    assert first == {"fallback": True}
    second = await model.ainvoke_structured("hello again", system="p", schema={"type": "object"})
    assert second == {"primary": True}
    assert client.models == [PRIMARY, PRIMARY, FALLBACK, PRIMARY]
    assert sleeps == [5.0]
