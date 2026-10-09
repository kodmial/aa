"""P0 #295: semantic selection over broad candidates, full text, multi-turn.

Covers the approved new behavior (not the former top-5/500-char clamps):

- a decisive passage at fused rank >5 and >16 is discovered, selected
  and used via the bounded semantic layer;
- essential information after character 500 reaches both the generator
  and the semantic verifier untruncated;
- multi-turn follow-ups preserve context and source coverage;
- planner accepts a flexible 1..16 useful-query count;
- broad retrieval keeps deep ranks inspectable within real budgets.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aa.conversation.planner_schema import (
    QueryPlan,
    QueryPlanValidationError,
    validate_query_plan,
)
from aa.conversation.prompt_builder import EvidencePassage, render_turn_context
from aa.conversation.response_units import split_response_units
from aa.conversation.semantic_selection import (
    MAX_SELECTED_CHUNKS,
    CandidatePreview,
    SemanticSelectionError,
    heuristic_select,
    order_pack_semantically,
    preview_candidates,
    rerank_previews_semantically,
    selection_telemetry,
    validate_semantic_selection,
)
from aa.conversation.turn_pipeline import run_v2_answer_turn
from aa.conversation.verifier import build_single_unit_text
from aa.retrieval.evidence import (
    MAX_PER_SECTION,
    POOL_CAP,
    TOP_CHILD_CAP,
    RetrievalConfig,
)


def _pack_entry(passage_id: str, text: str) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": passage_id.split("#")[0],
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _previews(count: int = 24, decisive_rank: int = 20) -> list[CandidatePreview]:
    previews: list[CandidatePreview] = []
    for rank in range(count):
        if rank == decisive_rank:
            text = (
                "Обычное введение. "
                + ("Фоновый контекст рядом. " * 10)
                + "Решающий признак: трезвая поддержка и утренний разбор тяги. "
                + ("Дополнительный контекст после. " * 10)
            )
        else:
            text = f"Фоновый отрывок {rank}. Общие слова без решающего признака. "
        previews.append(
            CandidatePreview(
                chunk_id=f"chapter-{(rank % 4) + 1}#c{rank:04d}",
                fused_rank=rank,
                fused_score=float(count - rank),
                section=f"chapter-{(rank % 4) + 1}",
                source_id="ru-fourth-edition-txt",
                preview_text=text[:400],
                full_text=text,
            )
        )
    return previews


def test_planner_accepts_flexible_useful_counts() -> None:
    for count in (1, 3, 7, 12, 16):
        plan = validate_query_plan(
            QueryPlan(
                mode="retrieval",
                resolved_intent="разобрать тягу утром",
                queries=[f"запрос {i}" for i in range(count)],
            )
        )
        assert len(plan.queries) == count
    with pytest.raises(QueryPlanValidationError):
        validate_query_plan(
            QueryPlan(
                mode="retrieval",
                resolved_intent="x",
                queries=[f"q{i}" for i in range(17)],
            )
        )


def test_broad_pool_keeps_deep_ranks_inspectable() -> None:
    assert POOL_CAP >= 64
    assert TOP_CHILD_CAP > 16
    assert MAX_PER_SECTION >= 4
    config = RetrievalConfig()
    assert config.pool_cap == POOL_CAP
    assert config.top_child_cap == TOP_CHILD_CAP


def test_semantic_rerank_promotes_deep_relevant_preview() -> None:
    previews = _previews()
    reranked = rerank_previews_semantically(
        previews, resolved_intent="трезвый утренний разбор тяги поддержка"
    )
    top_ids = [p.chunk_id for p in reranked[:MAX_SELECTED_CHUNKS]]
    decisive = previews[20].chunk_id
    assert decisive in top_ids
    # Nothing is dropped: every candidate stays inspectable.
    assert len(reranked) == len(previews)


def test_heuristic_selection_marks_deep_rank_telemetry() -> None:
    previews = _previews()
    selection = heuristic_select(previews, resolved_intent="трезвый утренний разбор тяги поддержка")
    assert previews[20].chunk_id in selection.selected_chunk_ids
    telemetry = selection_telemetry(previews=previews, selection=selection, latency_ms=3.0)
    assert telemetry["selection_deep_rank_gt5"] is True
    assert telemetry["selection_deep_rank_gt16"] is True
    assert telemetry["selection_max_rank"] > 16
    assert "selection_digest" in telemetry


def test_selection_rejects_unknown_ids_fail_closed() -> None:
    previews = _previews(count=8)
    known = {p.chunk_id for p in previews}
    with pytest.raises(SemanticSelectionError):
        validate_semantic_selection(
            {
                "selected_chunk_ids": ["chapter-9#c9999"],
                "need_more_detail": False,
                "followup_queries": [],
            },
            known_chunk_ids=known,
        )


def test_preview_discovery_only_full_text_fetch_separate() -> None:
    previews = preview_candidates(
        fused_ordered=[(f"id-{i}", float(10 - i)) for i in range(5)],
        texts={f"id-{i}": ("Полный текст отрывка. " * 50) for i in range(5)},
        sections={f"id-{i}": "chapter-1" for i in range(5)},
    )
    assert all(len(p.preview_text) <= 400 for p in previews)
    assert all(len(p.full_text) > 500 for p in previews)


def test_full_text_after_500_reaches_generator_and_verifier() -> None:
    prefix = "Вводный контекст. " * 30
    decisive_tail = "Решающее указание после пятисотого символа: утренний разбор."
    text = prefix + decisive_tail
    assert len(text) > 500
    assert len(prefix) > 500
    passage = EvidencePassage(passage_id="chapter-3#exp0020", source="s", section="c", text=text)
    prompt = render_turn_context(
        summary="", passages=[passage], user_message="Как разбирать тягу утром?"
    )
    assert decisive_tail in prompt
    data = [
        {
            "passage_id": passage.passage_id,
            "source_id": passage.source,
            "section_id": passage.section,
            "text": passage.text,
        }
    ]
    units = split_response_units("Понимаю. Разбираю тягу утром.")
    verifier_input = build_single_unit_text(
        unit=units[0],
        passages=data,
        resolved_intent="разобрать тягу утром",
        user_message="Как разбирать тягу утром?",
    )
    assert decisive_tail in verifier_input


async def test_deep_rank_pack_serves_through_full_turn() -> None:
    pack = [
        _pack_entry(f"chapter-{(i % 4) + 1}#exp{i:04d}", f"Фоновый отрывок {i}. Общие слова.")
        for i in range(19)
    ]
    decisive_text = (
        "Вводный фон. " * 20 + "Решающий ответ: трезвая поддержка и утренний разбор тяги."
    )
    pack.append(_pack_entry("chapter-3#exp0019", decisive_text))
    assert len(pack) == 20
    draft = "Трезвую поддержку рядом и утренний разбор тяги опишу подробно."

    class _Answer:
        def __init__(self) -> None:
            self.seen = 0

        async def ainvoke(self, messages: Any) -> AIMessage:
            final = str(messages[-1].content)
            self.seen = final.count("<passage ")
            assert decisive_text in final
            return AIMessage(content=draft)

    class _Verifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            assert decisive_text in prompt
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["p20"],
                "addresses_intent": True,
            }

    answer = _Answer()
    outcome = await run_v2_answer_turn(
        user_message="Как мне разбирать тягу утром?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=_Verifier(),
        planner_model=None,
        retrieval_index=None,
        resolved_intent="разобрать тягу утром трезвая поддержка",
    )
    assert answer.seen == len(pack)
    assert outcome["text"] == draft
    assert outcome["telemetry"]["answer_generation_window"] == len(pack)


async def test_multi_turn_followup_preserves_context_and_coverage() -> None:
    pack = [
        _pack_entry("chapter-1#exp0001", "Первый разговор про вечернюю тягу и поддержку рядом."),
        _pack_entry("chapter-3#exp0007", "Утренний разбор тяги и трезвая поддержка рядом."),
    ]
    # Semantic ordering must not drop any passage across turns.
    ordered = order_pack_semantically(
        pack,
        resolved_intent="а утром как разбирать",
        conversation_context="вечерняя тяга поддержка рядом",
    )
    assert {item["passage_id"] for item in ordered} == {item["passage_id"] for item in pack}
    draft = "Утром помогает трезвая поддержка и спокойный разбор тяги."

    class _Answer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            body = "\n".join(str(item.content) for item in messages)
            assert "а утром как разбирать" in body
            return AIMessage(content=draft)

    class _Verifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["p1"],
                "addresses_intent": True,
            }

    outcome = await run_v2_answer_turn(
        user_message="а утром как разбирать",
        summary="вечерняя тяга поддержка рядом",
        recent=[
            HumanMessage(content="вечером тяжело, что делать"),
            AIMessage(content="Поддержка рядом помогает пережить тягу."),
            HumanMessage(content="а утром как разбирать"),
        ],
        evidence_pack=pack,
        answer_model=_Answer(),
        verifier_model=_Verifier(),
        planner_model=None,
        retrieval_index=None,
        resolved_intent="утром разбирать тягу",
    )
    assert outcome["text"] == draft


def test_source_coverage_spans_sections_without_loss() -> None:
    pack = [
        _pack_entry(f"chapter-{i}#exp{i:04d}", f"Текст раздела {i} про поддержку и тягу.")
        for i in (1, 2, 3, 5, 7)
    ]
    ordered = order_pack_semantically(pack, resolved_intent="поддержка тяга")
    sections = {item["section_id"] for item in ordered}
    assert sections == {item["section_id"] for item in pack}
    assert len(sections) >= 4
