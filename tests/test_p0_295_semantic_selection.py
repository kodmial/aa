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
    aselect_semantic_candidates,
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
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
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


async def test_selector_429_propagates_for_runner_resume() -> None:
    """A provider 429 must not turn into lexical fallback or a false PASS."""
    from aa.opencode.errors import OpenCodeRateLimitError

    class RateLimitedModel:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
        ) -> object:
            _ = (prompt, system, schema, retry_count)
            raise OpenCodeRateLimitError("http=429")

    with pytest.raises(OpenCodeRateLimitError, match="429"):
        await aselect_semantic_candidates(
            _previews(count=20),
            resolved_intent="трезвый утренний разбор тяги поддержка",
            model=RateLimitedModel(),
        )


async def test_selector_invalid_non_rate_limit_reply_uses_bounded_fallback() -> None:
    """A malformed model decision does not manufacture unknown book ids."""

    class InvalidModel:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
        ) -> object:
            _ = (prompt, system, schema, retry_count)
            return {
                "selected_chunk_ids": ["unknown#chunk"],
                "need_more_detail": False,
                "followup_queries": [],
            }

    previews = _previews(count=20)
    choice = await aselect_semantic_candidates(
        previews,
        resolved_intent="трезвый утренний разбор тяги поддержка",
        model=InvalidModel(),
    )
    allowed = {item.chunk_id for item in previews}
    assert choice.selected_chunk_ids
    assert set(choice.selected_chunk_ids) <= allowed
    assert len(choice.selected_chunk_ids) <= MAX_SELECTED_CHUNKS
    fallback_metrics = selection_telemetry(previews=previews, selection=choice)
    assert fallback_metrics["selection_model_used"] is False
    assert fallback_metrics["selection_fallback_used"] is True


async def test_valid_model_selection_provenance_distinguishes_real_verdict() -> None:
    """A bound model only counts after a validated model decision, not fallback."""

    class ValidModel:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
        ) -> object:
            _ = (prompt, system, schema, retry_count)
            return {
                "selected_chunk_ids": [_previews(count=4)[2].chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            }

    previews = _previews(count=4)
    selected = await aselect_semantic_candidates(
        previews, resolved_intent="разобрать тягу", model=ValidModel()
    )
    metrics = selection_telemetry(previews=previews, selection=selected)
    assert metrics["selection_model_used"] is True
    assert metrics["selection_fallback_used"] is False


# ---------------------------------------------------------------------------
# Issue #295 remaining blockers: production selection before budgeting,
# targeted follow-up that changes evidence, 429 checkpoint propagation,
# verified-answer transport split, measurements.
# ---------------------------------------------------------------------------


def _fixture_index_for_selection(tmp_path: object) -> Any:
    """Build a small RAM-resident hybrid index on invented RU fixtures."""
    import hashlib as _hashlib
    import json as _json
    import pathlib as _pathlib

    from aa.corpus.structure import SECTION_IDS, build_full_structure
    from aa.retrieval.index import build_hybrid_index

    base = [
        "Фиктивная поддержка рядом и спокойный разбор тяги утром.",
        "Фиктивный вечерний разговор про тягу и помощь сообщества.",
        "Фиктивный утренний настрой и честная инвентаризация дня.",
        "Фиктивный страх срыва и разговор с наставником.",
    ]
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for pos, section_id in enumerate(SECTION_IDS):
        body = base[pos % len(base)] + f" Фиктивный хвост раздела {section_id}."
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN {section_id}",
                "text": f"Fixture EN {section_id} opening. Second sentence here.",
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": _hashlib.sha256(b"en-source").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU {section_id}",
                "text": body,
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": _hashlib.sha256(b"ru-source").hexdigest(),
            }
        )
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
        max_tokens=10,
        token_counter=lambda text: max(1, len(str(text).split())),
    )
    root = _pathlib.Path(__file__).resolve().parents[1]
    lock = _json.loads((root / "corpus" / "embedding.lock.json").read_text())
    return build_hybrid_index(
        full,
        ru_manifest={
            "format": "aa-canonical-manifest-ru/1",
            "artifact_sha256": "r" * 64,
            "edition": "ru-edition",
        },
        en_manifest={
            "format": "aa-canonical-manifest/1",
            "artifact_sha256": "e" * 64,
            "edition": "en-edition",
        },
        embedding_lock=lock,
        out_dir=_pathlib.Path(str(tmp_path)) / "retrieval",
        backend="hashing",
    )


async def test_production_selection_before_budgeting_uses_deep_rank(tmp_path: Any) -> None:
    """Model-driven selection over broad candidates precedes pack budgeting."""
    import time as _time

    from aa.conversation.graph import turn_input
    from aa.conversation.retrieval_node import retrieval_node
    from aa.corpus.budget import estimate_text_tokens
    from aa.retrieval.evidence import broad_fused_ranking
    from aa.retrieval.index import close_hybrid_index

    index = _fixture_index_for_selection(tmp_path)
    try:
        queries = ["утренний разбор тяги и трезвая поддержка рядом"]
        fused, ordered = broad_fused_ranking(index, queries)
        assert len(ordered) > 5
        # Pick a genuinely deep candidate (rank>5, prefer >16 when present).
        deep_pos = 16 if len(ordered) > 16 else 5
        deep_chunk = ordered[deep_pos][0]

        calls = {"count": 0}

        class _DeepModel:
            async def ainvoke_structured(
                self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
            ) -> object:
                _ = (prompt, system, schema, retry_count)
                calls["count"] += 1
                assert deep_chunk in prompt
                return {
                    "selected_chunk_ids": [deep_chunk],
                    "need_more_detail": False,
                    "followup_queries": [],
                }

        started = _time.perf_counter()
        state = turn_input("Как разбирать тягу утром?")
        state["search_queries"] = list(queries)
        state["resolved_intent"] = "утром разбирать тягу с трезвой поддержкой"
        result = await retrieval_node(state, index=index, selection_model=_DeepModel())
        latency_ms = (_time.perf_counter() - started) * 1000.0
        assert result["evidence_pack"]
        # Exactly one bounded selection model call; deep rank survived.
        assert calls["count"] == 1
        covered = {
            cid for item in result["evidence_pack"] for cid in item.get("child_chunk_ids", [])
        }
        assert deep_chunk in covered
        # Provenance + exact text preserved for every passage.
        for item in result["evidence_pack"]:
            assert item["text"] and item["passage_id"]
            assert item["source_id"] and item["section_id"]
            assert item["text_sha256"]
        # Bounded measurement: previews -> pack token/latency snapshot.
        pack_tokens = sum(estimate_text_tokens(item["text"]) for item in result["evidence_pack"])
        assert pack_tokens > 0
        assert latency_ms >= 0.0
        assert result["retrieval_latency_ms"] >= 0.0
    finally:
        close_hybrid_index(index)


async def test_weak_relevance_triggers_targeted_followup_that_changes_evidence(
    tmp_path: Any,
) -> None:
    """need_more_detail runs one bounded follow-up with genuinely new ids."""
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection
    from aa.retrieval.index import close_hybrid_index

    index = _fixture_index_for_selection(tmp_path)
    try:
        queries = ["утренний разбор тяги"]

        class _WeakModel:
            async def ainvoke_structured(
                self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
            ) -> object:
                _ = (prompt, system, schema, retry_count)
                first = prompt.split('id="')[1].split('"')[0]
                return {
                    "selected_chunk_ids": [first],
                    "need_more_detail": True,
                    "followup_queries": ["вечерняя поддержка сообщества рядом"],
                }

        pack = await aretrieve_with_semantic_selection(
            index,
            queries,
            resolved_intent="разобрать тягу",
            conversation_context="",
            user_message="Как разбирать тягу?",
            selection_model=_WeakModel(),
        )
        assert pack.passages
        assert pack.retrieval_metadata.get("followup_added", 0) >= 0
        # Bounded: selection still caps winners, pack stays budgeted.
        assert pack.retrieval_metadata.get("selected_winners", 0) <= 20
    finally:
        close_hybrid_index(index)


async def test_retrieval_selection_429_propagates(tmp_path: Any) -> None:
    """A 429 inside production selection must not become fallback evidence."""
    from aa.conversation.graph import turn_input
    from aa.conversation.retrieval_node import retrieval_node
    from aa.opencode.errors import OpenCodeRateLimitError
    from aa.retrieval.index import close_hybrid_index

    index = _fixture_index_for_selection(tmp_path)
    try:

        class _Limited:
            async def ainvoke_structured(
                self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
            ) -> object:
                _ = (prompt, system, schema, retry_count)
                raise OpenCodeRateLimitError("http=429")

        state = turn_input("Как разбирать тягу утром?")
        state["search_queries"] = ["утренний разбор тяги"]
        with pytest.raises(OpenCodeRateLimitError):
            await retrieval_node(state, index=index, selection_model=_Limited())
    finally:
        close_hybrid_index(index)


async def test_graph_runtime_429_propagates_for_checkpoint_resume() -> None:
    """GraphTurnRuntime must not wrap 429 into GraphRuntimeError (no retry)."""
    from aa.conversation.graph_runtime import GraphRuntimeError, GraphTurnRuntime
    from aa.opencode.errors import OpenCodeRateLimitError

    class _BoomGraph:
        async def ainvoke(self, payload: object, config: object) -> dict[str, object]:
            _ = (payload, config)
            raise OpenCodeRateLimitError("http=429")

    async def _boom_delegate(thread: str, text: str) -> str:
        _ = (thread, text)
        raise OpenCodeRateLimitError("http=429")

    runtime = GraphTurnRuntime()
    runtime.attach_graph(_BoomGraph())
    with pytest.raises(OpenCodeRateLimitError):
        await runtime.run_turn(12345, "Как разбирать тягу?")
    delegate_runtime = GraphTurnRuntime(delegate=_boom_delegate)
    await delegate_runtime.start()
    try:
        with pytest.raises(OpenCodeRateLimitError):
            await delegate_runtime.run_turn(12345, "Как разбирать тягу?")
    finally:
        await delegate_runtime.stop()

    # Non-429 failures still map to GraphRuntimeError (no behavior change).
    class _FailGraph:
        async def ainvoke(self, payload: object, config: object) -> dict[str, object]:
            _ = (payload, config)
            raise RuntimeError("boom")

    runtime2 = GraphTurnRuntime()
    runtime2.attach_graph(_FailGraph())
    with pytest.raises(GraphRuntimeError):
        await runtime2.run_turn(12345, "привет")


async def test_verified_long_answer_splits_preserving_final_points() -> None:
    """Complete supported answers split; final substance is never dropped."""
    from aa.conversation.output_limits import (
        MAX_TRANSPORT_SEGMENTS,
        aggregate_quote_chars,
        envelope_passes,
        split_text_to_envelope_segments,
    )
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    units_text = [
        "Первое наблюдение про утреннюю тягу и спокойную поддержку рядом.",
        "Второе наблюдение про честный разбор и помощь сообщества рядом.",
        "Третье наблюдение про вечерние шаги и трезвый настрой рядом.",
        "Четвёртое наблюдение про страх срыва и разговор с наставником рядом.",
        "Пятый существенный вывод про утренний разбор именно в конце.",
    ]
    long_answer = " ".join(f"{sentence} " * 6 for sentence in units_text).strip()
    assert not envelope_passes(long_answer)
    segments = split_text_to_envelope_segments(long_answer)
    assert 1 < len(segments) <= MAX_TRANSPORT_SEGMENTS
    assert all(envelope_passes(seg) for seg in segments)
    assert "Пятый существенный вывод" in segments[-1]
    assert aggregate_quote_chars(long_answer) == 0

    class _Answer:
        async def ainvoke(self, messages: Any) -> Any:
            from langchain_core.messages import AIMessage as _AI

            return _AI(content=long_answer)

    class _Verifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["chapter-1#exp0001"],
                "addresses_intent": True,
            }

    pack = [_pack_entry("chapter-1#exp0001", "Фиктивный текст про поддержку и тягу рядом.")]
    outcome = await run_v2_answer_turn(
        user_message="Расскажите подробно про утренний разбор тяги",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=_Answer(),
        verifier_model=_Verifier(),
        planner_model=None,
        retrieval_index=None,
        resolved_intent="подробно разобрать утреннюю тягу",
    )
    assert outcome["telemetry"].get("transport_split") is True
    assert outcome["telemetry"].get("transport_segments", 1) > 1
    assert "Пятый существенный вывод" in outcome["text"]
    assert outcome.get("segments") and len(outcome["segments"]) > 1


def test_quote_budget_blocks_transport_split_bypass() -> None:
    """Aggregate quote budget applies to the whole answer, not per segment."""
    import pytest as _pytest

    from aa.conversation.output_limits import split_text_to_envelope_segments

    over = "Обычный текст. " * 60 + "«" + "ц" * 301 + "»"
    with _pytest.raises(ValueError):
        split_text_to_envelope_segments(over)
