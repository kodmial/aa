"""P0 kodmial/aa#217 recurrence 2: converge Gate C+E in one cycle.

Live evidence on exact main a7d76f1 (run 37670332968) after the 0202b0b
input-window repair (answer window 6 + planner 500 chars):

- C:live-answer-no-generic-collapse on live-production-path with 3
  generic clarifications, and
- E:latency-budget-exceeded (p50 25.2s / p95 33.6s / max 44.7s over the
  30s budget, planner p50 3.6s / p95 9.2s / max 10.2s, retrieval p50
  0.4s, total p50 25.2s, repair_turns=0).

The prior repair saved only ~4s p50 (29.6s to 25.2s): planner improved
5.7s to 3.6s but total stayed 10s over the p95 SLO because the dominant
persistent cost is the sequential provider chain (answer text on strong
primary plus one concurrent verifier call per draft unit) with the
planner stuck on the weak fallback (served planner=space-bunny while
answer/verifier/summarizer served muse-spark). Input-count bounding
cannot close that gap, so this recurrence changes strategy at the
responsible boundaries instead of repeating it:

- planner transport decoupling (logical aa-planner-v2, wire omitted)
  serves the strong primary in one call instead of custom-selector 403
  plus 5s sleep plus omitted attempt plus weak fallback, cutting planner
  tail and improving pack relevance so fewer drafts collapse; and
- verifier fan-out bound (leading MAX_VERIFIED_UNITS) caps provider
  calls per turn at the answer-to-verifier handoff (output boundary),
  discarding the tail before any verifier call. Grounding stays strict
  (only validated supported units served).

Turn-independent, never an exact-question special case, Product
Contract #110 unchanged.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aa.conversation.model_adapter import (
    PLANNER_AGENT_V2,
    PLANNER_TRANSPORT_AGENT_V2,
    build_planner_model,
    clear_primary_circuit,
)
from aa.conversation.response_units import split_response_units
from aa.conversation.turn_pipeline import (
    MAX_VERIFIED_UNITS,
    cap_draft_to_verified_window,
    run_v2_answer_turn,
)
from aa.opencode.client import SessionInfo

MUSE = "opencode/muse-spark-1.3-contributor-free"
BUNNY = "opencode/space-bunny-free"


def _pack_entry(index: int) -> dict[str, Any]:
    body = f"Фиктивная поддержка рядом помогает. Отрывок {index}."
    return {
        "passage_id": f"chapter-3#exp{index:04d}",
        "text": body,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(body),
        "text_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


class _DecouplingClient:
    def __init__(self) -> None:
        self.wire_agents: list[str] = []
        self.audit_agents: list[str] = []
        self.models: list[str] = []

    async def create_session(self, title: str = "") -> SessionInfo:
        return SessionInfo(id="ses_1", title=title)

    async def delete_session(self, session_id: str) -> bool:
        return True

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        agent: str = "",
        system: str = "",
        audit_agent: str = "",
        **_: Any,
    ) -> str:
        self.wire_agents.append(agent)
        self.audit_agents.append(audit_agent)
        self.models.append(model)
        return "ok"

    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        agent: str = "",
        system: str = "",
        audit_agent: str = "",
        **_: Any,
    ) -> dict[str, object]:
        self.wire_agents.append(agent)
        self.audit_agents.append(audit_agent)
        self.models.append(model)
        return {"queries": ["запрос"] * 12}


def test_planner_factory_is_decoupled_with_fallback() -> None:
    """Decoupled planner keeps logical audit, omits wire, keeps fallback."""
    clear_primary_circuit()
    try:
        planner = build_planner_model(object(), primary_model=MUSE, fallback_model=BUNNY)  # type: ignore[arg-type]
        assert planner.agent == PLANNER_AGENT_V2 == "aa-planner-v2"
        assert planner.primary_model == MUSE
        assert planner.fallback_model == BUNNY
        assert planner.wire_agent == PLANNER_TRANSPORT_AGENT_V2 == ""
    finally:
        clear_primary_circuit()


async def test_planner_wire_omits_selector_but_keeps_logical_audit() -> None:
    """Planner wire omits the custom selector; audit keeps aa-planner-v2."""
    clear_primary_circuit()
    try:
        client = _DecouplingClient()
        planner = build_planner_model(client, primary_model=MUSE, fallback_model=BUNNY)  # type: ignore[arg-type]
        await planner._ainvoke_text("hello", system="planner-system")
        assert client.wire_agents == [""]
        assert client.audit_agents == ["aa-planner-v2"]
        assert client.models == [MUSE]
    finally:
        clear_primary_circuit()


async def test_planner_structured_wire_omits_selector() -> None:
    """Structured planner calls decouple the same way as text calls."""
    clear_primary_circuit()
    try:
        client = _DecouplingClient()
        planner = build_planner_model(client, primary_model=MUSE, fallback_model=BUNNY)  # type: ignore[arg-type]
        await planner.ainvoke_structured("hello", system="p", schema={"type": "object"})
        assert client.wire_agents == [""]
        assert client.audit_agents == ["aa-planner-v2"]
    finally:
        clear_primary_circuit()


def test_production_planner_uses_decoupled_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production wiring builds the planner decoupled with fallback kept."""
    import aa.conversation.graph as graph_module
    import aa.conversation.graph_runtime as runtime_module
    import aa.conversation.model_adapter as adapter_module

    seen: dict[str, Any] = {}

    class _DummyModel:
        def __init__(
            self,
            client: object,
            *,
            agent: str,
            primary_model: str,
            fallback_model: str = "",
            request_timeout: float = 120.0,
            transport_agent: str | None = None,
        ) -> None:
            self.client = client
            self.agent = agent
            self.primary_model = primary_model
            self.fallback_model = fallback_model
            self.request_timeout = request_timeout
            self.transport_agent = transport_agent

        def with_agent(self, agent: str) -> _DummyModel:
            return _DummyModel(
                self.client,
                agent=agent,
                primary_model=self.primary_model,
                fallback_model=self.fallback_model,
                request_timeout=self.request_timeout,
            )

        @property
        def wire_agent(self) -> str:
            if self.transport_agent is None:
                return self.agent
            return self.transport_agent

    def _fake_planner(
        client: object, *, primary_model: str, fallback_model: str, request_timeout: float = 120.0
    ) -> _DummyModel:
        seen["primary"] = primary_model
        seen["fallback"] = fallback_model
        return _DummyModel(
            client,
            agent="aa-planner-v2",
            primary_model=primary_model,
            fallback_model=fallback_model,
            request_timeout=request_timeout,
            transport_agent="",
        )

    def _fake_verifier(
        client: object, *, primary_model: str, request_timeout: float = 120.0
    ) -> _DummyModel:
        return _DummyModel(
            client,
            agent="aa-verifier-v2",
            primary_model=primary_model,
            fallback_model="",
            request_timeout=request_timeout,
            transport_agent="",
        )

    monkeypatch.setattr(adapter_module, "OpenCodeChatModel", _DummyModel)
    monkeypatch.setattr(adapter_module, "build_verifier_model", _fake_verifier)
    monkeypatch.setattr(adapter_module, "build_planner_model", _fake_planner)
    monkeypatch.setattr(graph_module, "build_turn_graph", lambda **kwargs: kwargs)
    runtime = runtime_module._ProductionGraphRuntime(
        client=object(),
        settings=SimpleNamespace(opencode_model=MUSE, opencode_fallback_model=BUNNY),
        index=None,
    )
    wired = runtime._build_graph(object())
    planner = wired["planner_model"]
    assert planner.wire_agent == ""
    assert planner.agent == "aa-planner-v2"
    assert planner.fallback_model == BUNNY
    # Derived text agents keep the custom wire (proven for text).
    assert wired["answer_model"].wire_agent == "aa-v2"
    assert seen["primary"] == MUSE


def test_verified_window_caps_long_drafts() -> None:
    """Five-sentence drafts verify only the leading window."""
    assert MAX_VERIFIED_UNITS == 3
    draft = " ".join(f"Поддержка рядом помогает спокойно {idx}." for idx in range(5))
    units = split_response_units(draft)
    assert len(units) == 5
    capped = cap_draft_to_verified_window(draft)
    capped_units = split_response_units(capped)
    assert len(capped_units) == MAX_VERIFIED_UNITS
    assert capped_units[0].text == units[0].text
    assert capped_units[-1].text == units[MAX_VERIFIED_UNITS - 1].text


def test_verified_window_passes_short_drafts_through() -> None:
    """Drafts at or below the cap travel untouched."""
    draft = "Поддержка рядом помогает пережить тягу спокойно."
    assert cap_draft_to_verified_window(draft) == draft
    assert cap_draft_to_verified_window("") == ""


class _AnswerModel:
    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        return AIMessage(content=self._drafts.pop(0))


class _CountingVerifier:
    def __init__(self, pack: list[dict[str, Any]]) -> None:
        self._pack = pack
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        return {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["p1"],
        }


async def test_long_draft_verifies_bounded_units_only() -> None:
    """A five-unit draft pays at most three verifier calls and serves prefix."""
    pack = [_pack_entry(index) for index in range(6)]
    long_draft = " ".join(f"Поддержка рядом помогает спокойно {idx}." for idx in range(5))
    answer = _AnswerModel([long_draft])
    verifier = _CountingVerifier(pack)
    outcome = await run_v2_answer_turn(
        user_message="что помогает при тяге?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
    )
    assert answer.calls == 1
    assert verifier.calls == MAX_VERIFIED_UNITS
    assert outcome["telemetry"]["verifier_units_cap"] == MAX_VERIFIED_UNITS
    served_units = split_response_units(outcome["text"])
    assert len(served_units) <= MAX_VERIFIED_UNITS
