"""P0 breaker kodmial/aa#290: unavailable-unit delivery narrowing.

Convergence scope ``gate:C`` with fingerprint
``C:live-meta-direct-1:live-production-path`` (local recurrence 3,
systemic recurrence 3, one turn with an unavailable verifier unit over
48 response units, verifier p95 51s / max 65s).

Architecture invariant: one verifier-unavailable unit is synthesized as
book-scoped unsupported and poisoned whole-turn telemetry (repair
skipped, relevance-padding rescue blocked by its ``not unavailable``
guard, conversational fallback withheld by the claims-book guard which
counted unavailable book labels as evidence, adequacy
unavailable-verifier fail) even when a verified relevant supported
subset existed. Whichever family hit the transient collapsed to a
generic retry/clarification while clean siblings passed, moving the
fingerprint across SHAs and paraphrases with no convergence.

Repair at the delivery boundary only: the narrowed subset excludes
unavailable units by construction (model verdicts ``supported`` plus
explicit ``addresses_intent`` for book units), and its telemetry
describes the served subset (zero unavailable, passed outcome) rather
than the discarded draft. No extra model call (Gate E SLO preserved),
per-claim grounding holds for exactly what is delivered, and turns with
no relevant supported book unit still fail closed. Generic coverage
only: invented fixture text, no literal qualification prompts.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

from aa.opencode.errors import OpenCodeTransientError

_FIXTURE_PASSAGE = "Изобретение спокойного вечернего распорядка помогает сегодня."
_RELEVANT = "Изобретение спокойного вечернего распорядка помогает сегодня."
_GLUE = "Понимаю, расскажите чуть подробнее."
_MIXED_DRAFT = f"{_RELEVANT} Дополнительная несвязанная мысль."


def _pack_entry() -> dict[str, Any]:
    return {
        "passage_id": "chapter-3#exp0000",
        "text": _FIXTURE_PASSAGE,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(_FIXTURE_PASSAGE),
        "text_sha256": hashlib.sha256(_FIXTURE_PASSAGE.encode("utf-8")).hexdigest(),
    }


def _unit_from_prompt(prompt: str) -> str:
    import re as _re

    match = _re.search(r"<response_unit>(.*?)</response_unit>", prompt, _re.S)
    return match.group(1).strip() if match else prompt


class _RelevantPlusUnavailableVerifier:
    """One relevant supported book unit plus one transport failure."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        if _unit_from_prompt(prompt) == _RELEVANT:
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["p1"],
                "addresses_intent": True,
            }
        raise OpenCodeTransientError("fixture transient verifier failure")

    async def _ainvoke_text(self, text: str, *, system: str | None = None) -> str:
        _ = (text, system)
        raise OpenCodeTransientError("fixture transient verifier failure")


class _GluePlusUnavailableVerifier:
    """One supported glue unit plus one transport failure."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        if _unit_from_prompt(prompt) == _GLUE:
            return {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            }
        raise OpenCodeTransientError("fixture transient verifier failure")

    async def _ainvoke_text(self, text: str, *, system: str | None = None) -> str:
        _ = (text, system)
        raise OpenCodeTransientError("fixture transient verifier failure")


class _StaticAnswer:
    def __init__(self, text: str) -> None:
        self._text = text

    async def ainvoke(self, messages: Any) -> Any:
        from langchain_core.messages import AIMessage

        _ = messages
        return AIMessage(content=self._text)


async def test_substantive_unavailable_narrows_to_relevant_subset() -> None:
    from aa.conversation.turn_pipeline import (
        run_v2_answer_turn,
    )

    outcome = await run_v2_answer_turn(
        user_message="Нейтральный вопрос о вечерней поддержке сегодня?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_StaticAnswer(_MIXED_DRAFT),
        verifier_model=_RelevantPlusUnavailableVerifier(),
        initial_query_count=12,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="Нейтральный вопрос о вечерней поддержке сегодня?",
    )
    from aa.conversation.failures import is_service_error as _ise290

    assert outcome["text"] == _RELEVANT
    assert not _ise290(outcome["text"])
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "narrowed-adequacy"
    assert telemetry["verifier_unavailable_units"] == 0
    assert telemetry["verifier_outcome"] == "passed"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["qualified"] is True


async def test_conversational_partial_unavailable_serves_fallback() -> None:
    from aa.conversation.turn_pipeline import (
        run_v2_answer_turn,
    )
    from aa.qualification.product_contract_live import _is_direct_meta_reply

    outcome = await run_v2_answer_turn(
        user_message="Нейтральное приветствие и вопрос о возможностях сегодня",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=_StaticAnswer(f"{_GLUE} Дополнительная мысль."),
        verifier_model=_GluePlusUnavailableVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=0,
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
    )
    from aa.conversation.failures import is_service_error as _ise290b

    assert outcome["text"] == f"{_GLUE} Дополнительная мысль."
    assert not _ise290b(outcome["text"])
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "conversational-generated"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["qualified"] is True
    assert _is_direct_meta_reply(outcome["text"], snapshot=dict(telemetry))


async def test_substantive_glue_only_with_unavailable_still_fails_closed() -> None:
    # No relevant supported book unit exists, so the turn must not serve
    # glue as a substantive success even though one glue unit verified.
    import pytest as _pt290

    from aa.conversation.failures import TurnFailed as _TF290
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    with _pt290.raises(_TF290):
        await run_v2_answer_turn(
            user_message="Нейтральный вопрос о вечерней поддержке сегодня?",
            summary="",
            recent=[],
            evidence_pack=[_pack_entry()],
            answer_model=_StaticAnswer(f"{_GLUE} Дополнительная мысль."),
            verifier_model=_GluePlusUnavailableVerifier(),
            initial_query_count=12,
            planner_reason="substantive-with-queries",
            planner_mode="retrieval",
            resolved_intent="Нейтральный вопрос о вечерней поддержке сегодня?",
        )
    return


def test_no_literal_qualification_branches() -> None:
    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "conversation"
        / "turn_pipeline.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "Чем ты вообще можешь быть полезен",
        "Слушай, а ты тут вообще чем помочь можешь",
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
        "Дома снова ссора",
        "Поругались дома из-за бухла",
        "покупать акции",
        "Какой телефон сейчас выгоднее купить",
        "Изобретение спокойного вечернего распорядка",
    ):
        assert fragment not in source
