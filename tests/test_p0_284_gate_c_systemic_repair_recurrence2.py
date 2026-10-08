"""P0 kodmial/aa#284 systemic recurrence 2: relevance-padding rescue.

Systemic failure class (Gate C ``live-production-path`` across distinct
main SHAs and scenario families): a substantive draft that mixes one
verifier-supported relevant book unit with verifier-supported but
irrelevant padding fails the whole turn (``answer_relevant`` false,
adequacy ``irrelevant-citation``) while a clean sibling passes, so the
fingerprint keeps moving across runs while narrow per-fingerprint
repairs never converge. The prior systemic repair (conversational
fallback) covers only planner-certified conversational turns; the
repair loop replans and regenerates but the narrowing granularity
stayed support-only, so padding survived every rescue and the turn
collapsed to a generic retry.

Architecture-level repair at the delivery boundary: after the bounded
repair loop, deterministically narrow to the relevant supported subset
using only the per-unit model verdicts (supported plus an explicit
``addresses_intent`` true verdict for book units; supported non-book
glue is kept) and serve it when the subset itself passes the envelope,
quote-budget, outbound-safety and whole-turn adequacy gates. No extra
model call (Gate E SLO preserved), per-claim grounding holds for
exactly what is delivered, and turns with no relevant supported book
unit still fail closed. Generic coverage only: no literal
qualification prompt is used below.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

from langchain_core.messages import AIMessage

_REQUEST = "Вечером не могу успокоиться, что может помочь?"
_PASSAGE = "Спокойный вечерний распорядок и поддержка рядом помогают справиться с беспокойством."
_RELEVANT = "Спокойный вечерний распорядок и поддержка рядом помогают справиться с беспокойством."
_PADDING = "Регулярная уборка помогает держать дом в порядке."
_MIXED_DRAFT = f"{_RELEVANT} {_PADDING}"


def _pack_entry() -> dict[str, Any]:
    return {
        "passage_id": "chapter-3#exp0000",
        "text": _PASSAGE,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(_PASSAGE),
        "text_sha256": hashlib.sha256(_PASSAGE.encode("utf-8")).hexdigest(),
    }


class _MixedDraftAnswer:
    """Serve one fixed two-sentence draft mixing relevant guidance with padding."""

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        return AIMessage(content=_MIXED_DRAFT)


class _PaddingAwareVerifier:
    """Model-driven verdicts: relevant book unit relevant, padding irrelevant.

    Both units are verifier-supported with passage provenance; only the
    relevance verdict differs, which is exactly the systemic class: the
    turn is grounded but padded, so the whole-turn aggregate is
    irrelevant while a relevant subset exists.
    """

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        import re as _re

        _ = (system, schema, retry_count)
        match = _re.search(r"<response_unit>(.*?)</response_unit>", prompt, _re.S)
        unit = match.group(1).strip() if match else prompt
        if "уборка" in unit:
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": False,
            }
        return {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["chapter-3#exp0000"],
            "addresses_intent": True,
        }


class _AllIrrelevantVerifier(_PaddingAwareVerifier):
    """Every supported book unit is irrelevant: no relevant subset exists."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["chapter-3#exp0000"],
            "addresses_intent": False,
        }


async def _run_turn(verifier: Any) -> dict[str, Any]:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    return await run_v2_answer_turn(
        user_message=_REQUEST,
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_MixedDraftAnswer(),
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent=_REQUEST,
    )


async def test_mixed_relevant_padding_serves_relevant_subset() -> None:
    outcome = await _run_turn(_PaddingAwareVerifier())
    telemetry = dict(outcome["telemetry"])
    assert outcome["text"] == _RELEVANT
    assert _PADDING not in outcome["text"]
    assert telemetry["answer_outcome"] == "narrowed-adequacy"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["qualified"] is True


async def test_mixed_subset_passes_qualification_gates() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import (
        _is_grounded_substantive_reply,
        assess_reply_relevance_with_rubric,
    )

    outcome = await run_v2_answer_turn(
        user_message=_REQUEST,
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_MixedDraftAnswer(),
        verifier_model=_PaddingAwareVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent=_REQUEST,
    )
    telemetry = dict(outcome["telemetry"])
    snapshot = {
        "answer_outcome": str(telemetry.get("answer_outcome", "unknown")),
        "verifier_outcome": str(telemetry.get("verifier_outcome", "unknown")),
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 12,
        "retrieval_passages": 1,
        "verified_book_units": 1,
        "response_units": 1,
        "adequacy_verdict": str(telemetry.get("adequacy_verdict", "unknown")),
        "failure_category": str(telemetry.get("failure_category", "")),
        "planner_reason": "substantive-with-queries",
        "answers_request": bool(telemetry.get("answers_request", False)),
        "technically_grounded": bool(telemetry.get("technically_grounded", False)),
        "qualified": bool(telemetry.get("qualified", False)),
    }
    reply = str(outcome["text"])
    assert assess_reply_relevance_with_rubric(_REQUEST, reply, telemetry=snapshot) is True
    assert _is_grounded_substantive_reply(snapshot, reply) is True


async def test_all_irrelevant_padding_still_fails_closed() -> None:
    from aa.conversation.turn_pipeline import NATURAL_RETRY_VARIANTS, select_retry_reply

    outcome = await _run_turn(_AllIrrelevantVerifier())
    telemetry = dict(outcome["telemetry"])
    assert outcome["text"] == select_retry_reply(_REQUEST)
    assert outcome["text"] in set(NATURAL_RETRY_VARIANTS)
    assert telemetry["adequacy_verdict"] == "fail"
    assert telemetry["answers_request"] is False


def test_relevant_narrowing_helpers_use_model_verdicts_only() -> None:
    from aa.conversation.response_units import ResponseUnitDraft
    from aa.conversation.turn_pipeline import (
        has_relevant_supported_book_unit,
        keep_relevant_supported_text,
        keep_relevant_supported_units,
        narrowed_grounding_state,
    )
    from aa.conversation.verifier_schema import GroundingResult, UnitVerdict

    units = [
        ResponseUnitDraft(unit_id="u1", text=_RELEVANT, char_start=0, char_end=len(_RELEVANT)),
        ResponseUnitDraft(unit_id="u2", text=_PADDING, char_start=0, char_end=len(_PADDING)),
        ResponseUnitDraft(unit_id="u3", text="Понимаю, это непросто.", char_start=0, char_end=1),
    ]
    result = GroundingResult(
        verified=True,
        units=[
            UnitVerdict(
                unit_id="u1",
                scope="book",
                supported=True,
                evidence_passage_ids=["chapter-3#exp0000"],
                addresses_intent=True,
            ),
            UnitVerdict(
                unit_id="u2",
                scope="book",
                supported=True,
                evidence_passage_ids=["chapter-3#exp0000"],
                addresses_intent=False,
            ),
            UnitVerdict(
                unit_id="u3",
                scope="conversation_glue",
                supported=True,
                evidence_passage_ids=[],
                addresses_intent=False,
            ),
        ],
        all_required_supported=True,
        answer_relevant=False,
        relevance_category="irrelevant-citation",
    )
    kept = keep_relevant_supported_units(units, result)
    assert [unit.unit_id for unit in kept] == ["u1", "u3"]
    assert keep_relevant_supported_text(units, result) == f"{_RELEVANT} Понимаю, это непросто."
    assert has_relevant_supported_book_unit(result) is True
    state = narrowed_grounding_state(kept, result)
    assert state["answer_relevant"] is True
    assert state["all_required_supported"] is True


def test_no_exact_question_branches() -> None:
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
    ):
        assert fragment not in source


def test_no_domain_keyword_tables_in_repair() -> None:
    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "conversation"
        / "turn_pipeline.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "_RECOVERY_DOMAIN_STEMS",
        "_STEP_WORD_RE",
        "_GREETING_VOCABULARY",
        "тянет выпить",
        "тянеет выпить",
    ):
        assert fragment not in source
