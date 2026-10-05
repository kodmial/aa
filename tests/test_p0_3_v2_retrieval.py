"""P0-3 target retrieval pipeline tests (issue #116).

Covers the Definition of Done on invented fixture text (no canonical
book text committed):

- multi-query hybrid retrieval wired to the #113 planner query list and
  the #115 RAM-resident index (10..16 queries, both branches each);
- global RRF (k=60) with per-query retention, overlap dedup/diversity;
- RRF-only selection (fused-score order, no second-stage reranker);
- small-to-big expansion with merge semantics;
- compact exact-Russian Evidence Pack under the 16k token budget;
- RRF-only benchmark (no recall regression, zero hot-path disk reads);
- zero legacy handcrafted semantic logic in the new path;
- zero BGE/FlagEmbedding/cross-encoder runtime in production.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
import sqlite3
from typing import Any

import pytest

from aa.conversation.prompt_builder import EvidencePassage
from aa.corpus.budget import RETRIEVED_PASSAGES_BUDGET_TOKENS, estimate_text_tokens
from aa.corpus.structure import SECTION_IDS, build_full_structure
from aa.retrieval import evidence as evidence_mod
from aa.retrieval.evidence import (
    POOL_CAP,
    TOP_CHILD_CAP,
    EvidenceError,
    EvidencePassageData,
    RetrievalConfig,
    render_book_evidence,
    retrieve_evidence,
    select_passages_under_budget,
    to_prompt_passages,
)
from aa.retrieval.fusion import RRF_K
from aa.retrieval.index import HybridIndex, build_hybrid_index, close_hybrid_index

ROOT = pathlib.Path(__file__).resolve().parents[1]

RU_FIXTURES: dict[str, str] = {
    "doctors-opinion": (
        "Фиктивное мнение доктора о тяге и одержимости. Наблюдение продолжается.\n\n"
        "Второй абзац мнения доктора. Выводы записываются аккуратно и честно."
    ),
    "chapter-1": (
        "Фиктивный рассказ о первом глотке. Герой начал бухать каждый вечер.\n\n"
        "Второй абзац рассказа. Утро после выпивки было тяжелым и долгим."
    ),
    "chapter-2": (
        "Фиктивный выход есть для пьющих. Надежда и поддержка рядом всегда.\n\n"
        "Второй абзац выхода. Сообщество встречает новичков тепло и спокойно."
    ),
    "chapter-3": (
        "Фиктивный алкоголизм как феномен тяги. Тяга к алкоголю приходит внезапно. "
        "Сорвался после долгой трезвости.\n\n"
        "Второй абзац об алкоголизме. Рецидив начинается с первой рюмки."
    ),
    "chapter-4": (
        "Фиктивные размышления агностика про астролябию сомнений. "
        "Готовность принять помощь растет.\n\n"
        "Второй абзац агностика. Сомнения обсуждаются открыто и честно."
    ),
    "chapter-5": (
        "Фиктивная программа в действии требует честности. Инвентаризация "
        "обид и страхов начинается сегодня.\n\n"
        "Второй абзац программы. Обида разбирается честно и спокойно."
    ),
    "chapter-6": (
        "Фиктивная работа по шагам продолжается. Утром делаем инвентаризацию. "
        "Молитва утром и медитация вечером помогают.\n\n"
        "Второй абзац работы. Вечером подводим итоги дня честно."
    ),
    "chapter-7": (
        "Фиктивная работа с другими людьми. Несем весть тем кто страдает. "
        "Помогаем другому алкоголику бескорыстно.\n\n"
        "Второй абзац помощи. Разговор ведется спокойно и честно."
    ),
    "chapter-8": (
        "Фиктивная жинка ругает из-за пьянки. Женушка переживает за семью.\n\n"
        "Второй абзац о семье. Пьянство разрушает доверие постепенно."
    ),
    "chapter-9": (
        "Фиктивные новые отношения в семье. Доверие возвращается постепенно.\n\n"
        "Второй абзац семьи. Разговоры становятся спокойнее и теплее."
    ),
    "chapter-10": (
        "Фиктивное обращение к работодателям. Трезвость на рабочем месте важна.\n\n"
        "Второй абзац работодателям. Поддержка коллег помогает многим."
    ),
    "chapter-11": (
        "Фиктивный взгляд в будущее сообщества. Содружество растет. "
        "Членство в сообществе открыто для всех.\n\n"
        "Второй абзац будущего. Планы строятся на трезвую голову."
    ),
}


def _word_counter(text: str) -> int:
    return max(1, len(text.split()))


def _build_index(tmp_path: pathlib.Path, **kwargs: Any) -> HybridIndex:
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": (
                    f"Fixture EN {section_id} opening sentence. Second sentence here.\n\n"
                    f"Fixture EN {section_id} second paragraph here."
                ),
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": hashlib.sha256(b"en-source").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU TITLE {section_id}",
                "text": RU_FIXTURES[section_id],
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": hashlib.sha256(b"ru-source").hexdigest(),
            }
        )
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
        max_tokens=int(kwargs.pop("max_tokens", 10)),
        token_counter=_word_counter,
    )
    lock = json.loads((ROOT / "corpus" / "embedding.lock.json").read_text())
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
        out_dir=tmp_path / "retrieval",
        backend="hashing",
        **kwargs,
    )


def _twelve_queries(base: str = "тяга к алкоголю") -> list[str]:
    return [f"{base} вариант {index}" for index in range(12)]


# ---------------------------------------------------------------------------
# Planner-query execution contract: every query runs both branches.
# ---------------------------------------------------------------------------


def test_all_queries_execute_both_branches(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        queries = _twelve_queries()
        lexical_calls: list[dict[str, Any]] = []
        dense_calls: list[dict[str, Any]] = []
        from aa.retrieval.lexical import lexical_search_conn as _real_lexical

        real_search = index.dense.search

        def _count_lexical(
            connection: sqlite3.Connection, query: str, *, top_k: int = 40
        ) -> list[tuple[str, float]]:
            lexical_calls.append({"query": query, "top_k": top_k})
            return _real_lexical(connection, query, top_k=top_k)

        def _count_dense(query: list[float], *, top_k: int) -> list[tuple[str, float]]:
            dense_calls.append({"top_k": top_k})
            return real_search(query, top_k=top_k)

        import unittest.mock as _mock

        with (
            _mock.patch("aa.retrieval.evidence.lexical_search_conn", _count_lexical),
            _mock.patch.object(index.dense, "search", _count_dense),
        ):
            pack = retrieve_evidence(index, queries)
        assert len(lexical_calls) == len(queries) == 12
        assert len(dense_calls) == len(queries) == 12
        assert {call["query"] for call in lexical_calls} == set(queries)
        assert {call["top_k"] for call in lexical_calls} == {40}
        # Dense depth is capped by the corpus size on tiny fixtures.
        assert {call["top_k"] for call in dense_calls} == {min(40, index.chunk_count)}
        assert pack.retrieval_metadata["branch_lists"] == 2 * len(queries) == 24
        assert pack.retrieval_metadata["branch_top_k"] == 40
    finally:
        close_hybrid_index(index)


def test_fused_order_determines_selection(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        queries = _twelve_queries()
        pack = retrieve_evidence(index, queries)
        assert pack.passages
        metadata = pack.retrieval_metadata
        assert metadata["retrieval_backend"] == "rrf-only/1"
        assert metadata["primary_query_digest"] != ""
        assert "reranker_model" not in metadata
        assert "reranker_revision" not in metadata
        assert "reranker_backend" not in metadata
        assert "reranker_pool_cap" not in metadata
    finally:
        close_hybrid_index(index)


def test_rrf_receives_all_branch_rankings(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        queries = _twelve_queries()
        seen: dict[str, Any] = {}
        from aa.retrieval.fusion import rrf_fuse as _real_fuse

        def _watch(
            ranked_lists: list[list[tuple[str, float]]], *, k: int = RRF_K
        ) -> dict[str, Any]:
            seen["lists"] = len(ranked_lists)
            seen["k"] = k
            seen["non_empty"] = sum(1 for item in ranked_lists if item)
            fused = _real_fuse(ranked_lists, k=k)
            return dict(fused)

        import unittest.mock as _mock

        with _mock.patch("aa.retrieval.evidence.rrf_fuse", _watch):
            retrieve_evidence(index, queries)
        assert seen["lists"] == 2 * len(queries) == 24
        assert seen["k"] == RRF_K == 60
        assert seen["non_empty"] == seen["lists"]
    finally:
        close_hybrid_index(index)


def test_empty_planner_result_performs_no_retrieval(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        import unittest.mock as _mock

        with (
            _mock.patch.object(
                evidence_mod,
                "lexical_search_conn",
                side_effect=AssertionError("no lexical on empty plan"),
            ),
            _mock.patch.object(
                index.dense, "search", side_effect=AssertionError("no dense on empty plan")
            ),
        ):
            pack = retrieve_evidence(index, [])
        assert pack.passages == ()
        assert pack.total_tokens == 0
        assert pack.retrieval_metadata["retrieval_skipped"] is True
    finally:
        close_hybrid_index(index)


@pytest.mark.parametrize("count", [1, 9, 17])
def test_invalid_query_counts_fail_closed(tmp_path: pathlib.Path, count: int) -> None:
    index = _build_index(tmp_path)
    try:
        with pytest.raises(EvidenceError):
            retrieve_evidence(
                index,
                [f"запрос {pos}" for pos in range(count)],
            )
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# Fusion/dedup/diversity: collapse exact and overlapping child spans.
# ---------------------------------------------------------------------------


def test_duplicate_and_overlapping_children_collapse(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        pack = retrieve_evidence(index, _twelve_queries())
        assert pack.passages
        child_ids = [cid for item in pack.passages for cid in item.child_chunk_ids]
        assert len(set(child_ids)) == len(child_ids)
        spans = [(item.section_id, item.char_start, item.char_end) for item in pack.passages]
        for pos in range(len(spans)):
            for other in range(pos + 1, len(spans)):
                if spans[pos][0] != spans[other][0]:
                    continue
                assert spans[pos][2] <= spans[other][1] or spans[other][2] <= spans[pos][1]
    finally:
        close_hybrid_index(index)


def test_per_query_best_candidate_retained(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        # The nonsense-unique term only occurs in chapter-4; only the query
        # carrying it can contribute that chunk.
        queries = ["астролябия сомнений"] + _twelve_queries("фон трезвости")[:11]
        wide = RetrievalConfig(
            pool_cap=POOL_CAP,
            top_child_cap=POOL_CAP,
            budget_tokens=RETRIEVED_PASSAGES_BUDGET_TOKENS,
        )
        pack = retrieve_evidence(index, queries, config=wide)
        sections = {item.section_id for item in pack.passages}
        assert "chapter-4" in sections
    finally:
        close_hybrid_index(index)


def test_small_pool_preserves_global_rrf_depth() -> None:
    """Per-query bests are retained first; the remainder fills by global RRF.

    With 2 queries and a 4-cap pool both uniques are retained plus the
    globally top-ranked candidates. With 12 queries and a 16-cap pool
    every distinct per-query best fits and must be retained; the
    64-cap pool still retains every per-query best.
    """
    from aa.retrieval.evidence import fuse_query_pool

    ranked_lists = [
        [("g1", 9.0), ("g2", 8.0), ("g3", 7.0), ("g4", 6.0), (f"u{pos}", 1.0)] for pos in range(2)
    ]
    per_query_ids = [[f"u{pos}"] for pos in range(2)]
    fused, pool_ids = fuse_query_pool(ranked_lists, per_query_ids, pool_cap=4)
    assert len(pool_ids) == 4
    assert "u0" in pool_ids
    assert "u1" in pool_ids
    assert "g1" in pool_ids
    assert "g2" in pool_ids
    # Review trigger: 12 queries, 16-cap pool, >8 distinct uniques.
    trigger_ranked = [[("g1", 9.0), ("g2", 8.0), (f"u{pos}", 1.0)] for pos in range(12)]
    trigger_per_query = [[f"u{pos}"] for pos in range(12)]
    _, trigger_pool = fuse_query_pool(trigger_ranked, trigger_per_query, pool_cap=16)
    assert len(trigger_pool) == 14
    for pos in range(12):
        assert f"u{pos}" in trigger_pool
    assert "g1" in trigger_pool
    assert "g2" in trigger_pool
    _, full_pool_ids = fuse_query_pool(ranked_lists, per_query_ids, pool_cap=POOL_CAP)
    for chunk_id in ("u0", "u1", "g1", "g2", "g3", "g4"):
        assert chunk_id in full_pool_ids


# ---------------------------------------------------------------------------
# RRF-only selection: fused order honored, no reranker.
# ---------------------------------------------------------------------------


def test_fused_selection_ordering_is_honored(tmp_path: pathlib.Path) -> None:
    from aa.retrieval.evidence import (
        dedup_and_diversify,
        fuse_query_pool,
        run_branch_searches,
        select_top_candidates,
    )

    index = _build_index(tmp_path)
    try:
        queries = _twelve_queries("молитва утром")
        ranked, per_query = run_branch_searches(index, queries)
        fused, pool_ids = fuse_query_pool(ranked, per_query, pool_cap=POOL_CAP)
        diverse = dedup_and_diversify(index, pool_ids, fused, pool_cap=POOL_CAP)
        assert len(diverse) >= 2
        scored = sorted(diverse, key=lambda item: item.fused_score, reverse=True)
        winners = select_top_candidates(scored, top_cap=TOP_CHILD_CAP)
        assert [item.chunk_id for item in winners] == [
            item.chunk_id for item in scored[:TOP_CHILD_CAP]
        ]
        # End to end: the top-fused chunk lands in the first passage.
        pack = retrieve_evidence(
            index,
            queries,
            config=RetrievalConfig(
                pool_cap=POOL_CAP,
                top_child_cap=len(scored),
            ),
        )
        first = pack.passages[0]
        assert scored[0].chunk_id in first.child_chunk_ids
    finally:
        close_hybrid_index(index)


def test_pool_and_top_caps_configurable(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        config = RetrievalConfig(pool_cap=8, top_child_cap=4)
        pack = retrieve_evidence(index, _twelve_queries(), config=config)
        assert pack.retrieval_metadata["pool_cap"] == 8
        assert pack.retrieval_metadata["top_child_cap"] == 4
        assert pack.retrieval_metadata["diverse_unique"] <= 8
        assert pack.retrieval_metadata["selected_winners"] <= 4
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# Small-to-big expansion: coherent exact spans, merge, section boundaries.
# ---------------------------------------------------------------------------


def test_parent_neighbor_expansion_returns_exact_spans(tmp_path: pathlib.Path) -> None:
    from aa.retrieval.evidence import expand_small_to_big
    from aa.retrieval.fusion import FusedCandidate

    index = _build_index(tmp_path)
    try:
        by_parent: dict[str, list[str]] = {}
        for chunk_id, record in index.chunks.items():
            by_parent.setdefault(record.parent, []).append(chunk_id)
        parent = next(key for key, ids in by_parent.items() if len(ids) >= 2)
        members = sorted(by_parent[parent])
        winners = [
            FusedCandidate(
                chunk_id=members[0],
                fused_score=2.0 + len(members) - pos,
                lexical_rank=1,
                dense_rank=1,
                lexical_score=0.0,
                dense_score=1.0,
            )
            for pos in range(len(members))
        ]
        passages = expand_small_to_big(index, winners)
        assert passages
        covered = {cid for passage in passages for cid in passage.child_chunk_ids}
        for chunk_id in members:
            assert chunk_id in covered
        for passage in passages:
            ordered = sorted(
                passage.child_chunk_ids,
                key=lambda cid: (
                    index.chunks[cid].char_start,
                    index.chunks[cid].char_end,
                    cid,
                ),
            )
            # No synthetic glue: contiguous tiling concatenates exactly.
            assert passage.exact_text == "".join(index.chunks[cid].text for cid in ordered)
            for first_id, second_id in zip(ordered, ordered[1:], strict=False):
                assert index.chunks[second_id].char_start == index.chunks[first_id].char_end
            first = index.chunks[passage.child_chunk_ids[0]]
            assert passage.section_id == first.section
            assert passage.source_id == first.source_id
            assert passage.char_start == min(
                index.chunks[cid].char_start for cid in passage.child_chunk_ids
            )
            assert passage.char_end == max(
                index.chunks[cid].char_end for cid in passage.child_chunk_ids
            )
            assert (
                passage.text_sha256
                == hashlib.sha256(passage.exact_text.encode("utf-8")).hexdigest()
            )
        passage = next(item for item in passages if members[0] in item.child_chunk_ids)
    finally:
        close_hybrid_index(index)


def test_overlapping_expansions_merge(tmp_path: pathlib.Path) -> None:
    from aa.retrieval.evidence import expand_small_to_big
    from aa.retrieval.fusion import FusedCandidate

    index = _build_index(tmp_path)
    try:
        by_parent: dict[str, list[str]] = {}
        for chunk_id, record in index.chunks.items():
            by_parent.setdefault(record.parent, []).append(chunk_id)
        parent = next(key for key, ids in by_parent.items() if len(ids) >= 2)
        members = sorted(by_parent[parent])[:2]
        winners = [
            FusedCandidate(
                chunk_id=cid,
                fused_score=2.0 - pos,
                lexical_rank=1,
                dense_rank=None,
                lexical_score=0.0,
                dense_score=None,
            )
            for pos, cid in enumerate(members)
        ]
        passages = expand_small_to_big(index, winners, neighbor_window=0)
        assert passages
        covered = {cid for passage in passages for cid in passage.child_chunk_ids}
        assert set(members) <= covered
        # Contiguous siblings share one exact passage (no synthetic glue).
        holder = next(item for item in passages if members[0] in item.child_chunk_ids)
        assert members[1] in holder.child_chunk_ids
        assert holder.exact_text == "".join(
            index.chunks[cid].text
            for cid in sorted(
                holder.child_chunk_ids,
                key=lambda cid: (
                    index.chunks[cid].char_start,
                    index.chunks[cid].char_end,
                    cid,
                ),
            )
        )
    finally:
        close_hybrid_index(index)


def test_expansion_stays_within_section(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        pack = retrieve_evidence(index, _twelve_queries())
        for passage in pack.passages:
            sections = {index.chunks[cid].section for cid in passage.child_chunk_ids}
            assert sections == {passage.section_id}
            sources = {index.chunks[cid].source_id for cid in passage.child_chunk_ids}
            assert sources == {passage.source_id}
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# Evidence budget: atomic selection, never silently truncated.
# ---------------------------------------------------------------------------


def test_evidence_budget_is_atomic(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        full = retrieve_evidence(index, _twelve_queries())
        assert full.passages
        tiny = RetrievalConfig(budget_tokens=40)
        pack = retrieve_evidence(index, _twelve_queries(), config=tiny)
        assert pack.total_tokens <= 40
        for passage in pack.passages:
            assert estimate_text_tokens(passage.exact_text) <= 40
            # No mid-chunk truncation: at least one full child chunk survives.
            assert any(
                index.chunks[cid].text in passage.exact_text for cid in passage.child_chunk_ids
            )
            assert (
                passage.text_sha256
                == hashlib.sha256(passage.exact_text.encode("utf-8")).hexdigest()
            )
    finally:
        close_hybrid_index(index)


def test_budget_fallback_preserves_skipped_rrf_winner(
    tmp_path: pathlib.Path,
) -> None:
    index = _build_index(tmp_path)
    try:
        first, second = list(index.chunks.values())[:2]
        first_passage = EvidencePassageData(
            passage_id="first-expanded",
            exact_text=first.text,
            source_id=first.source_id,
            section_id=first.section,
            child_chunk_ids=(first.chunk_id,),
            char_start=first.char_start,
            char_end=first.char_end,
            text_sha256=first.text_sha256,
            source_sha256=first.source_sha256,
        )
        oversized_text = second.text * 20
        oversized = EvidencePassageData(
            passage_id="second-expanded",
            exact_text=oversized_text,
            source_id=second.source_id,
            section_id=second.section,
            child_chunk_ids=(second.chunk_id,),
            char_start=second.char_start,
            char_end=second.char_end,
            text_sha256=hashlib.sha256(oversized_text.encode("utf-8")).hexdigest(),
            source_sha256=second.source_sha256,
        )
        budget = estimate_text_tokens(first.text) + estimate_text_tokens(second.text)
        selected, total = select_passages_under_budget(
            [first_passage, oversized],
            budget_tokens=budget,
            index=index,
            priority_child_ids=(first.chunk_id, second.chunk_id),
        )
        assert total <= budget
        by_child = {passage.child_chunk_ids[0]: passage for passage in selected}
        assert first.chunk_id in by_child
        assert second.chunk_id in by_child
        assert by_child[second.chunk_id].exact_text == second.text
        assert by_child[second.chunk_id].text_sha256 == second.text_sha256
    finally:
        close_hybrid_index(index)


def test_budget_fallback_reserves_priority_before_later_passages(
    tmp_path: pathlib.Path,
) -> None:
    index = _build_index(tmp_path)
    try:
        first, second = list(index.chunks.values())[:2]
        oversized_text = first.text * 20
        oversized_first = EvidencePassageData(
            passage_id="first-expanded",
            exact_text=oversized_text,
            source_id=first.source_id,
            section_id=first.section,
            child_chunk_ids=(first.chunk_id,),
            char_start=first.char_start,
            char_end=first.char_end,
            text_sha256=hashlib.sha256(oversized_text.encode("utf-8")).hexdigest(),
            source_sha256=first.source_sha256,
        )
        later = EvidencePassageData(
            passage_id="later-full",
            exact_text=second.text,
            source_id=second.source_id,
            section_id=second.section,
            child_chunk_ids=(second.chunk_id,),
            char_start=second.char_start,
            char_end=second.char_end,
            text_sha256=second.text_sha256,
            source_sha256=second.source_sha256,
        )
        first_need = estimate_text_tokens(first.text)
        second_need = estimate_text_tokens(second.text)
        budget = max(first_need, second_need)
        selected, total = select_passages_under_budget(
            [oversized_first, later],
            budget_tokens=budget,
            index=index,
            priority_child_ids=(first.chunk_id, second.chunk_id),
        )
        assert total <= budget
        child_ids = [passage.child_chunk_ids[0] for passage in selected]
        assert first.chunk_id in child_ids
        assert child_ids[0] == first.chunk_id
        assert selected[0].exact_text == first.text
    finally:
        close_hybrid_index(index)

# ---------------------------------------------------------------------------
# Evidence Pack contract: exact text plus minimal provenance only.
# ---------------------------------------------------------------------------


def test_ranking_metadata_excluded_from_book_evidence(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        pack = retrieve_evidence(index, _twelve_queries())
        assert pack.passages
        rendered = render_book_evidence(pack)
        assert rendered.startswith("<book_evidence>")
        assert rendered.rstrip().endswith("</book_evidence>")
        lowered = rendered.casefold()
        for forbidden in (
            "fused_score",
            "lexical_rank",
            "dense_rank",
            "rerank",
            "rrf",
            "bm25",
            "embed",
            "preview",
            "planner",
            "score",
        ):
            assert forbidden not in lowered, forbidden
        for passage in pack.passages:
            assert passage.exact_text in rendered
            assert passage.passage_id in rendered
            assert passage.section_id in rendered
        prompt_passages = to_prompt_passages(pack)
        assert prompt_passages
        for item in prompt_passages:
            assert isinstance(item, EvidencePassage)
            assert set(item.__dict__) == {"passage_id", "source", "section", "text"}
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# New-path isolation: zero legacy semantic logic, zero BGE runtime.
# ---------------------------------------------------------------------------


def test_new_retrieval_path_has_no_legacy_imports() -> None:
    package = ROOT / "src" / "aa"
    modules = [
        package / "retrieval" / "evidence.py",
        package / "conversation" / "retrieval_node.py",
        package / "qualification" / "v2_retrieval.py",
    ]
    legacy_modules = {
        "aa.retrieval.planner",
        "aa.conversation.orchestrator",
        "aa.conversation.meta",
    }
    forbidden_symbols = (
        "_SLANG_EXPANSIONS",
        "_THEME_MARKERS",
        "_BROAD_COVERAGE_QUERIES",
        "ru-query-plan-v1",
        "aspect_search_queries",
        "search_aspect",
        "search_plan",
        "validate_plan",
        "is_substantive",
    )
    for path in modules:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in legacy_modules, f"{path.name}: {node.module}"
                for alias in node.names:
                    assert alias.name not in {
                        "aspect_search_queries",
                        "validate_plan",
                        "search_plan",
                        "search_aspect",
                    }, f"{path.name}: {alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in legacy_modules, f"{path.name}: {alias.name}"
        for symbol in forbidden_symbols:
            # Word-boundary match: "validate_plan" must not collide with the
            # new path's own "validate_planner_queries".
            assert re.search(r"\b" + re.escape(symbol) + r"\b", source) is None, (
                f"{path.name}: {symbol}"
            )


def test_production_path_has_no_bge_runtime() -> None:
    package = ROOT / "src" / "aa"
    modules = [
        package / "retrieval" / "evidence.py",
        package / "conversation" / "retrieval_node.py",
        package / "conversation" / "graph.py",
        package / "qualification" / "v2_retrieval.py",
    ]
    for path in modules:
        source = path.read_text(encoding="utf-8")
        lowered = source.casefold()
        assert "crossencoderreranker" not in lowered, path.name
        assert "flagreranker" not in lowered, path.name
        assert "flagembedding" not in lowered, path.name
        assert "bge-reranker" not in lowered, path.name
        assert "get_reranker" not in source, path.name
        assert "reranker.lock.json" not in source, path.name
    assert not (package / "retrieval" / "reranker.py").exists()
    assert not (ROOT / "corpus" / "reranker.lock.json").exists()
    assert not (ROOT / "scripts" / "prefetch_reranker.py").exists()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "FlagEmbedding" not in pyproject


def test_hot_path_performs_zero_disk_reads(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _build_index(tmp_path)
    try:
        import pathlib as _pathlib
        import sqlite3 as _sqlite3

        def _blocked_connect(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("disk must not be opened on the hot path")

        def _blocked_read_text(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("disk read must not occur on the hot path")

        def _blocked_read_bytes(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("disk read must not occur on the hot path")

        monkeypatch.setattr(_sqlite3, "connect", _blocked_connect)
        monkeypatch.setattr(_pathlib.Path, "read_text", _blocked_read_text)
        monkeypatch.setattr(_pathlib.Path, "read_bytes", _blocked_read_bytes)
        queries = _twelve_queries("трезвость поддержка")
        pack = retrieve_evidence(index, queries)
        assert pack.passages
        assert render_book_evidence(pack).startswith("<book_evidence>")
        assert to_prompt_passages(pack)
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# RRF-only benchmark: recall, determinism, stage monotonicity.
# ---------------------------------------------------------------------------


def _gold_index(tmp_path: pathlib.Path) -> Any:
    from aa.qualification.aa_retrieval import build_fixture_full

    full = build_fixture_full(max_tokens=256)
    lock = json.loads((ROOT / "corpus" / "embedding.lock.json").read_text())
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
        out_dir=tmp_path / "retrieval-gold",
        backend="hashing",
    )


def test_benchmark_rrf_only_no_regression(tmp_path: pathlib.Path) -> None:
    from aa.qualification.aa_retrieval import RECALL_GATE_AT_5, load_gold
    from aa.qualification.v2_retrieval import (
        duplicate_query_rate,
        planner_proxy_queries,
        query_diversity,
        run_v2_case,
        summarize_v2,
        v2_quality_gate,
    )

    index = _gold_index(tmp_path)
    try:
        cases = load_gold(ROOT / "qualification" / "aa-retrieval.gold.v1.json")
        assert len(cases) >= 20
        for case in cases:
            queries = planner_proxy_queries(case)
            assert 10 <= len(queries) <= 16
            assert 0.0 <= query_diversity(queries) <= 1.0
            assert 0.0 <= duplicate_query_rate(queries) <= 1.0
        results = [run_v2_case(index, case) for case in cases]
        summary = summarize_v2(results)
        # No regression in the existing qualified retrieval recall gate.
        assert summary.target_recall_at_5 >= RECALL_GATE_AT_5
        assert summary.duplicate_rate <= 0.05
        assert summary.mean_target_tokens <= RETRIEVED_PASSAGES_BUDGET_TOKENS
        assert summary.p95_target_latency_ms >= summary.p50_target_latency_ms >= 0.0
        assert summary.unsupported_clean_rate >= 0.0
        passed, _ = v2_quality_gate(summary)
        assert passed is True
        # Deterministic: same inputs produce the same pack sections.
        repeat = run_v2_case(index, cases[0])
        assert repeat.target_sections_10 == results[0].target_sections_10
        # Warm latency is measured warm (second run): it must stay within
        # the interactive budget on the RAM-only path.
        from aa.retrieval.evidence import INTERACTIVE_LATENCY_BUDGET_MS

        assert summary.p95_target_latency_ms <= INTERACTIVE_LATENCY_BUDGET_MS
    finally:
        close_hybrid_index(index)


def test_planner_to_retrieval_stage_recalls_monotonic(tmp_path: pathlib.Path) -> None:
    from aa.qualification.aa_retrieval import load_gold
    from aa.qualification.v2_retrieval import run_v2_case

    index = _gold_index(tmp_path)
    try:
        cases = [
            case
            for case in load_gold(ROOT / "qualification" / "aa-retrieval.gold.v1.json")
            if not case.is_unsupported
        ][:6]
        assert cases
        for case in cases:
            result = run_v2_case(index, case)
            # Pack winners come from the pool, pool candidates from the union.
            if result.pack_recall:
                assert result.pool_recall
            if result.pool_recall:
                assert result.union_recall
            assert result.selection_loss == (result.pool_recall and not result.pack_recall)
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# Graph wiring: planner queries to Evidence Pack state (RRF-only).
# ---------------------------------------------------------------------------


async def test_graph_retrieval_node_wires_evidence_pack(tmp_path: pathlib.Path) -> None:
    from langchain_core.runnables import RunnableLambda

    from aa.conversation.graph import build_turn_graph

    index = _build_index(tmp_path)
    try:
        plan = {"queries": _twelve_queries("срыв и тяга")}

        async def _plan(_messages: Any) -> Any:
            return plan

        graph = build_turn_graph(
            planner_model=RunnableLambda(_plan),
            retrieval_index=index,
        )
        from aa.conversation.graph import turn_input

        result = await graph.ainvoke(turn_input("тяга вечером"))
        assert result["planner_invoked"] is True
        assert len(result["search_queries"]) == 12
        assert result["evidence_pack"]
        assert result["retrieval_hits"]
        for entry in result["evidence_pack"]:
            assert entry["text"]
            assert entry["passage_id"] and entry["section_id"] and entry["source_id"]
            assert "fused_score" not in entry
            assert "rerank" not in " ".join(entry.keys()).casefold()
    finally:
        close_hybrid_index(index)


async def test_graph_without_index_keeps_stub(tmp_path: pathlib.Path) -> None:
    from langchain_core.runnables import RunnableLambda

    from aa.conversation.graph import build_turn_graph, turn_input

    async def _plan(_messages: Any) -> Any:
        return {"queries": []}

    graph = build_turn_graph(planner_model=RunnableLambda(_plan))
    result = await graph.ainvoke(turn_input("привет"))
    assert result["evidence_pack"] == []
    assert result["retrieval_hits"] == []


def test_make_retrieval_node_binds_directly(tmp_path: pathlib.Path) -> None:
    """The RRF-only node binds without performance acceptance gates."""
    from aa.conversation.retrieval_node import make_retrieval_node

    index = _build_index(tmp_path)
    try:
        node = make_retrieval_node(index=index)
        assert callable(node)
    finally:
        close_hybrid_index(index)


async def test_retrieval_node_propagates_latency_budget_signal(
    tmp_path: pathlib.Path,
) -> None:
    """RRF-only turns stay visible in state instead of looking interactive."""
    from aa.conversation.graph import turn_input
    from aa.conversation.retrieval_node import retrieval_node
    from aa.retrieval.evidence import INTERACTIVE_LATENCY_BUDGET_MS

    index = _build_index(tmp_path)
    try:
        state = turn_input("тяга вечером")
        state["search_queries"] = _twelve_queries("тяга поддержка")
        result = await retrieval_node(state, index=index)
        assert result["evidence_pack"]
        latency_ms = result["retrieval_latency_ms"]
        over_budget = result["retrieval_over_budget"]
        assert isinstance(latency_ms, float) and latency_ms >= 0.0
        assert over_budget == (latency_ms > INTERACTIVE_LATENCY_BUDGET_MS)
        # RAM-only RRF is fast: it must not read as over budget.
        assert over_budget is False
        skipped = turn_input("привет")
        skipped_result = await retrieval_node(skipped, index=index)
        assert skipped_result["retrieval_latency_ms"] == 0.0
        assert skipped_result["retrieval_over_budget"] is False
    finally:
        close_hybrid_index(index)


def test_rrf_only_quality_gate_passes_on_fixture(tmp_path: pathlib.Path) -> None:
    """The RRF-only gate passes on fixture data (no BGE latency block)."""
    from aa.qualification.aa_retrieval import load_gold
    from aa.qualification.v2_retrieval import (
        run_v2_case,
        summarize_v2,
        v2_production_status,
        v2_quality_gate,
    )

    index = _gold_index(tmp_path)
    try:
        cases = load_gold(ROOT / "qualification" / "aa-retrieval.gold.v1.json")[:8]
        results = [run_v2_case(index, case) for case in cases]
        summary = summarize_v2(results)
        passed, _ = v2_quality_gate(summary)
        assert passed is True
        status = v2_production_status(summary)
        assert status["quality_gate_passed"] is True
        assert status["cutover_allowed"] is True
        assert status["production_promotion_blocked"] is False
    finally:
        close_hybrid_index(index)
