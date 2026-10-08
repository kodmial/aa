"""P0 kodmial/aa#208 regression: repair Gate C live-path-failed.

Live evidence on exact main e92385f (run 37615447071) showed the
live-telegram-evidence lane failing with the concrete component signature:

- live-answer-no-generic-collapse (1 generic clarification of 16 turns),
- live-text-max-over-budget (max 64.0s over the 30s hard budget, p50 34.1s,
  p95 59.2s over 14 ordinary turns),

with stage outcomes verifier ``unsupported`` 10/12 and planner ``replanned``
10/12 while the planner served only the weak Space Bunny fallback and
answer/summarizer/verifier served strong Muse Spark. Every planner call
paid a doomed omitted-structured round-trip (custom 403 x2 plus 5s sleep
plus omitted deterministic failure plus weak fallback) before serving,
and each of the 10 repair re-plans repeated it, pushing slow turns over
budget while one turn narrowed to empty and clarified generically.

The repair has two general parts, neither an exact-question special case,
Product Contract #110 unchanged:

- omitted-structured capability cache: a deterministic omitted-structured
  failure marks this model path for a TTL, so later planner turns skip the
  doomed omitted attempt and go directly to the configured fallback,
  saving one slow provider round-trip per planner invocation including
  repair re-plans. Transient/timeout fall back once without poisoning the
  cache. Provider 429 always propagates and never marks the cache.
- turn repair time budget: when a turn already exceeds the live-SLO budget
  before repair, remaining repair rounds are skipped and the turn narrows
  to validated supported material instead of burning further sequential
  provider rounds. Grounding stays strict (only Pydantic-validated
  supported units are served, otherwise clarification); fast turns still
  use both repair rounds.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from aa.conversation import turn_pipeline as turn_pipeline_module
from aa.conversation.model_adapter import (
    OpenCodeChatModel,
    clear_omitted_structured_cache,
    clear_primary_circuit,
    omitted_structured_unavailable,
)
from aa.conversation.response_units import split_response_units
from aa.conversation.turn_pipeline import TURN_REPAIR_TIME_BUDGET_S, run_v2_answer_turn
from aa.conversation.verifier import clear_verifier_capability_cache
from aa.opencode.client import SessionInfo
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
)

PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"


@pytest.fixture(autouse=True)
def _clear_state() -> Any:
    clear_primary_circuit()
    clear_omitted_structured_cache()
    clear_verifier_capability_cache()
    yield
    clear_primary_circuit()
    clear_omitted_structured_cache()
    clear_verifier_capability_cache()


class _StructuredScriptedClient:
    """Scripted OpenCode client distinguishing custom vs omitted selector."""

    def __init__(self, *, structured_outcomes: list[object]) -> None:
        self._structured_outcomes = list(structured_outcomes)
        self.models: list[str] = []
        self.wire_agents: list[str] = []
        self._sessions = 0

    async def create_session(self, title: str = "") -> SessionInfo:
        self._sessions += 1
        return SessionInfo(id=f"ses_{self._sessions}", title=title)

    async def delete_session(self, session_id: str) -> bool:
        return True

    async def send_message(self, session_id: str, text: str, **_: Any) -> str:
        raise AssertionError("text path unused in this test")

    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        agent: str = "",
        **_: Any,
    ) -> dict[str, object]:
        self.models.append(model)
        self.wire_agents.append(agent)
        outcome = self._structured_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, dict)
        return outcome


async def test_omitted_structured_deterministic_skipped_on_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic omitted failure is skipped on the next planner turn."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client = _StructuredScriptedClient(
        structured_outcomes=[
            OpenCodeProviderAccessError("opencode provider access rejected: http=403"),
            OpenCodeProviderAccessError("opencode provider access rejected: http=403"),
            OpenCodeDeterministicError("opencode structured output missing"),
            {"queries": ["fallback query"] * 12},
            {"queries": ["fallback query"] * 12},
        ]
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    first = await model.ainvoke_structured("hello", system="p", schema={"type": "object"})
    assert first == {"queries": ["fallback query"] * 12}
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK]
    assert client.wire_agents == ["aa-planner-v2", "aa-planner-v2", "", "aa-planner-v2"]
    assert omitted_structured_unavailable(model) is True
    assert sleeps == [5.0]

    second = await model.ainvoke_structured("again", system="p", schema={"type": "object"})
    assert second == {"queries": ["fallback query"] * 12}
    # Circuit open plus cached omitted capability: direct fallback, no sleep,
    # no omitted PRIMARY attempt.
    assert client.models == [PRIMARY, PRIMARY, PRIMARY, FALLBACK, FALLBACK]
    assert client.wire_agents[-1] == "aa-planner-v2"
    assert sleeps == [5.0]


async def test_omitted_structured_429_never_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 429 on the omitted path propagates and never marks the cache."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client = _StructuredScriptedClient(
        structured_outcomes=[
            OpenCodeProviderAccessError("opencode provider access rejected: http=403"),
            OpenCodeProviderAccessError("opencode provider access rejected: http=403"),
            OpenCodeRateLimitError("opencode request rate-limited: http=429"),
        ]
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    with pytest.raises(OpenCodeRateLimitError):
        await model.ainvoke_structured("hello", system="p", schema={"type": "object"})
    assert omitted_structured_unavailable(model) is False
    assert FALLBACK not in client.models


def _pack_entry(text: str = "Фиктивная поддержка рядом помогает пережить тягу.") -> dict[str, Any]:
    return {
        "passage_id": "chapter-3#exp0000",
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class _AnswerModel:
    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)
        self.calls = 0

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls += 1
        if not self._drafts:
            raise AssertionError("answer model called more times than scripted")
        return AIMessage(content=self._drafts.pop(0))


class _VerifierModel:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results: list[dict[str, Any]] = []
        for entry in results:
            for unit in entry.get("units", []):
                supported = bool(unit.get("supported", False))
                self._results.append(
                    {
                        "requires_book_evidence": str(unit.get("scope", "book")) == "book",
                        "supported": supported,
                        "evidence_passage_ids": list(unit.get("evidence_passage_ids", [])),
                        "addresses_intent": bool(unit.get("addresses_intent", supported)),
                    }
                )
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        if not self._results:
            raise AssertionError("verifier called more times than scripted")
        return dict(self._results.pop(0))


class _PlannerModel:
    def __init__(self, plans: list[dict[str, Any]]) -> None:
        self._plans = list(plans)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        if not self._plans:
            raise AssertionError("planner called more times than scripted")
        return dict(self._plans.pop(0))


def _twelve_queries(base: str = "поддержка трезвость") -> list[str]:
    return [f"{base} вариант {index}" for index in range(12)]


async def test_repair_budget_exceeded_skips_repair_and_narrows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow turn skips repair and narrows instead of burning provider rounds."""
    pack = [_pack_entry()]
    good_text = "Поддержка рядом помогает пережить тягу спокойно."
    bad_text = "Луна лечит тягу за один вечер без усилий."
    good_units = split_response_units(good_text)
    assert len(good_units) == 1
    mixed_draft = f"{good_text} {bad_text}"
    mixed_units = split_response_units(mixed_draft)
    assert len(mixed_units) == 2
    answer = _AnswerModel([mixed_draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": mixed_units[0].unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    },
                    {
                        "unit_id": mixed_units[1].unit_id,
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    },
                ],
                "all_required_supported": False,
            }
        ]
    )
    planner = _PlannerModel(
        [
            {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries("вечер"),
            }
        ]
    )

    from aa.retrieval import evidence as evidence_mod

    def _fail_retrieve(index: Any, queries: object, *, config: Any = None) -> Any:
        raise AssertionError("repair retrieval must not run over budget")

    monkeypatch.setattr(evidence_mod, "retrieve_evidence", _fail_retrieve)
    monkeypatch.setattr(turn_pipeline_module, "TURN_REPAIR_TIME_BUDGET_S", -1.0)
    assert TURN_REPAIR_TIME_BUDGET_S > 0

    outcome = await run_v2_answer_turn(
        user_message="что помогает при тяге?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=planner,
        retrieval_index=object(),
    )
    assert outcome["rounds"] == 0
    assert planner.calls == 0
    assert outcome["telemetry"]["repair_budget_exceeded"] is True
    assert outcome["text"] != turn_pipeline_module.NATURAL_CLARIFICATION_REPLY
    assert good_text.split()[0] in outcome["text"]


async def test_fast_turn_still_repairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fast turn still uses the repair loop when under budget."""
    pack = [_pack_entry()]
    bad_draft = "Луна лечит тягу за один вечер без усилий."
    repaired_draft = "Поддержка рядом помогает пережить тягу спокойно."
    repaired_units = split_response_units(repaired_draft)
    assert len(repaired_units) == 1
    answer = _AnswerModel([bad_draft, repaired_draft])
    bad_units = split_response_units(bad_draft)
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                    for unit in bad_units
                ],
                "all_required_supported": False,
            },
            {
                "units": [
                    {
                        "unit_id": repaired_units[0].unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": ["chapter-3#extra"],
                    }
                ],
                "all_required_supported": True,
            },
        ]
    )
    planner = _PlannerModel(
        [
            {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries("вечер"),
            }
        ]
    )

    from aa.retrieval import evidence as evidence_mod
    from aa.retrieval.evidence import EvidencePack, EvidencePassageData

    def _fake_retrieve(index: Any, queries: object, *, config: Any = None) -> Any:
        text = "Фиктивная другая поддержка рядом."
        passage = EvidencePassageData(
            passage_id="chapter-3#extra",
            exact_text=text,
            source_id="ru-fourth-edition-txt",
            section_id="chapter-3",
            child_chunk_ids=("chapter-3:ru:9",),
            char_start=0,
            char_end=len(text),
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            source_sha256="s" * 64,
        )
        return EvidencePack(
            passages=(passage,), total_tokens=10, corpus_version="v", retrieval_metadata={}
        )

    monkeypatch.setattr(evidence_mod, "retrieve_evidence", _fake_retrieve)
    assert TURN_REPAIR_TIME_BUDGET_S > 0

    outcome = await run_v2_answer_turn(
        user_message="что помогает при тяге?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=planner,
        retrieval_index=object(),
    )
    assert outcome["rounds"] == 1
    assert outcome["text"] == repaired_draft
    assert outcome["telemetry"]["repair_budget_exceeded"] is False
