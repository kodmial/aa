"""P0-3 target retrieval pipeline tests (issue #116).

Covers the Definition of Done on invented fixture text (no canonical
book text committed):

- multi-query hybrid retrieval wired to the #113 planner query list and
  the #115 RAM-resident index (10..16 queries, both branches each);
- global RRF (k=60) with per-query retention, overlap dedup/diversity;
- pinned local BGE reranker loaded/reused (FlagEmbedding interface,
  immutable revision lock, offline hermetic backend for CI);
- small-to-big expansion with merge semantics;
- compact exact-Russian Evidence Pack under the 16k token budget;
- frozen benchmark: RRF-only baseline vs target (no recall regression,
  broad cases non-regressive, zero hot-path disk reads);
- zero legacy handcrafted semantic logic in the new path.
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
    POST_RERANK_CHILD_CAP,
    RERANKER_POOL_CAP,
    EvidenceError,
    RetrievalConfig,
    render_book_evidence,
    retrieve_evidence,
    to_prompt_passages,
)
from aa.retrieval.fusion import RRF_K
from aa.retrieval.index import HybridIndex, build_hybrid_index, close_hybrid_index
from aa.retrieval.reranker import (
    OFFLINE_BACKEND_NAME,
    RERANKER_MODEL_ID,
    CrossEncoderReranker,
    get_reranker,
    load_reranker_lock,
    reset_reranker_cache,
    verify_cached_reranker,
)

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


class _RecordingReranker(CrossEncoderReranker):
    """Deterministic stub reranker recording hot-path usage."""

    def __init__(self, lock: dict[str, Any], scores: list[float] | None = None) -> None:
        super().__init__(lock=lock, backend="test-recording/1")
        self.calls: list[tuple[str, int]] = []
        self._scores = scores

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, len(texts)))
        if self._scores is not None:
            assert len(self._scores) == len(texts)
            return list(self._scores)
        return [float(len(texts) - pos) for pos in range(len(texts))]


def _reranker_lock() -> dict[str, Any]:
    return load_reranker_lock(ROOT / "corpus" / "reranker.lock.json")


def _offline_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(lock=_reranker_lock(), backend=OFFLINE_BACKEND_NAME)


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
            pack = retrieve_evidence(index, queries, reranker=_offline_reranker())
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


def test_queries_zero_is_canonical_reranker_query(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        queries = ["прямая формулировка тяги"] + _twelve_queries("фон")[:11]
        recorder = _RecordingReranker(_reranker_lock())
        pack = retrieve_evidence(index, queries, reranker=recorder)
        assert pack.passages
        # One batched scoring call with the canonical query only.
        assert len(recorder.calls) == 1
        assert recorder.calls[0][0] == queries[0]
        assert recorder.calls[0][1] == pack.retrieval_metadata["diverse_unique"]
        assert pack.retrieval_metadata["reranker_query_digest"] != ""
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
            retrieve_evidence(index, queries, reranker=_offline_reranker())
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
            pack = retrieve_evidence(index, [], reranker=_offline_reranker())
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
                reranker=_offline_reranker(),
            )
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# Fusion/dedup/diversity: collapse exact and overlapping child spans.
# ---------------------------------------------------------------------------


def test_duplicate_and_overlapping_children_collapse(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        pack = retrieve_evidence(index, _twelve_queries(), reranker=_offline_reranker())
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
            reranker_pool_cap=RERANKER_POOL_CAP,
            post_rerank_child_cap=RERANKER_POOL_CAP,
            budget_tokens=RETRIEVED_PASSAGES_BUDGET_TOKENS,
        )
        pack = retrieve_evidence(index, queries, config=wide, reranker=_offline_reranker())
        sections = {item.section_id for item in pack.passages}
        assert "chapter-4" in sections
    finally:
        close_hybrid_index(index)


def test_small_pool_preserves_global_rrf_depth() -> None:
    """A small interactive pool keeps global RRF depth, not just per-query bests.

    With 4 queries and a 4-cap pool, uncapped per-query retention would fill
    the pool with per-query uniques and drop the globally top-ranked
    candidates. Retention is capped at half the pool so the interactive
    16-cap prefix keeps RRF-ranked depth; the frozen 64-cap pool still
    retains every per-query best.
    """
    from aa.retrieval.evidence import fuse_query_pool

    ranked_lists = [
        [("g1", 9.0), ("g2", 8.0), ("g3", 7.0), ("g4", 6.0), (f"u{pos}", 1.0)] for pos in range(4)
    ]
    per_query_ids = [[f"u{pos}"] for pos in range(4)]
    fused, pool_ids = fuse_query_pool(ranked_lists, per_query_ids, pool_cap=4)
    assert len(pool_ids) == 4
    assert "g1" in pool_ids
    assert "g2" in pool_ids
    _, full_pool_ids = fuse_query_pool(ranked_lists, per_query_ids, pool_cap=RERANKER_POOL_CAP)
    for chunk_id in ("u0", "u1", "u2", "u3", "g1", "g2", "g3", "g4"):
        assert chunk_id in full_pool_ids


# ---------------------------------------------------------------------------
# Reranker: ordering honored, canonical query, batched long-lived reuse.
# ---------------------------------------------------------------------------


def test_reranker_ordering_is_honored(tmp_path: pathlib.Path) -> None:
    from aa.retrieval.evidence import (
        dedup_and_diversify,
        fuse_query_pool,
        rerank_candidates,
        run_branch_searches,
    )

    index = _build_index(tmp_path)
    try:
        queries = _twelve_queries("молитва утром")
        ranked, per_query = run_branch_searches(index, queries)
        fused, pool_ids = fuse_query_pool(ranked, per_query, pool_cap=RERANKER_POOL_CAP)
        diverse = dedup_and_diversify(index, pool_ids, fused, pool_cap=RERANKER_POOL_CAP)
        assert len(diverse) >= 2
        # Reverse the fused order deterministically through stub scores.
        scored = sorted(diverse, key=lambda item: item.fused_score, reverse=True)
        stub_scores = [float(pos) for pos in range(len(scored))]
        stub = _RecordingReranker(_reranker_lock(), scores=stub_scores)
        winners, _ = rerank_candidates(index, scored, reranker_query=queries[0], reranker=stub)
        expected = list(reversed(scored))[:POST_RERANK_CHILD_CAP]
        assert [item.chunk_id for item in winners] == [item.chunk_id for item in expected]
        # End to end: the top-scored chunk lands in the first passage.
        # Pin the frozen full-quality pool so the full-depth ordering check
        # is unaffected by the interactive 16-cap library default.
        favor_last = _RecordingReranker(
            _reranker_lock(), scores=[float(pos) for pos in range(len(scored))]
        )
        pack = retrieve_evidence(
            index,
            queries,
            config=RetrievalConfig(
                reranker_pool_cap=RERANKER_POOL_CAP,
                post_rerank_child_cap=len(scored),
            ),
            reranker=favor_last,
        )
        first = pack.passages[0]
        assert scored[-1].chunk_id in first.child_chunk_ids
    finally:
        close_hybrid_index(index)


def test_reranker_pool_and_post_caps_configurable(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        config = RetrievalConfig(reranker_pool_cap=8, post_rerank_child_cap=4)
        pack = retrieve_evidence(
            index, _twelve_queries(), config=config, reranker=_offline_reranker()
        )
        assert pack.retrieval_metadata["reranker_pool_cap"] == 8
        assert pack.retrieval_metadata["post_rerank_child_cap"] == 4
        assert pack.retrieval_metadata["diverse_unique"] <= 8
        assert pack.retrieval_metadata["reranked_winners"] <= 4
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
        assert len(passages) == 1
        passage = passages[0]
        for chunk_id in members:
            assert index.chunks[chunk_id].text in passage.exact_text
        assert passage.exact_text == "\n".join(
            index.chunks[cid].text
            for cid in sorted(
                passage.child_chunk_ids,
                key=lambda cid: (
                    index.chunks[cid].char_start,
                    index.chunks[cid].char_end,
                    cid,
                ),
            )
        )
        first = index.chunks[passage.child_chunk_ids[0]]
        assert passage.section_id == first.section
        assert passage.source_id == first.source_id
        assert passage.char_start == min(
            index.chunks[cid].char_start for cid in passage.child_chunk_ids
        )
        assert passage.char_end == max(
            index.chunks[cid].char_end for cid in passage.child_chunk_ids
        )
        assert passage.text_sha256 == hashlib.sha256(passage.exact_text.encode("utf-8")).hexdigest()
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
        assert len(passages) == 1
        assert set(members) <= set(passages[0].child_chunk_ids)
    finally:
        close_hybrid_index(index)


def test_expansion_stays_within_section(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        pack = retrieve_evidence(index, _twelve_queries(), reranker=_offline_reranker())
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
        full = retrieve_evidence(index, _twelve_queries(), reranker=_offline_reranker())
        assert full.passages
        tiny = RetrievalConfig(budget_tokens=40)
        pack = retrieve_evidence(
            index, _twelve_queries(), config=tiny, reranker=_offline_reranker()
        )
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


# ---------------------------------------------------------------------------
# Evidence Pack contract: exact text plus minimal provenance only.
# ---------------------------------------------------------------------------


def test_ranking_metadata_excluded_from_book_evidence(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        pack = retrieve_evidence(index, _twelve_queries(), reranker=_offline_reranker())
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
# New-path isolation: zero legacy semantic logic.
# ---------------------------------------------------------------------------


def test_new_retrieval_path_has_no_legacy_imports() -> None:
    package = ROOT / "src" / "aa"
    modules = [
        package / "retrieval" / "evidence.py",
        package / "retrieval" / "reranker.py",
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


def test_hot_path_performs_zero_disk_reads(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _build_index(tmp_path)
    lock = _reranker_lock()
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
        pack = retrieve_evidence(
            index, queries, reranker=CrossEncoderReranker(lock=lock, backend="test/1")
        )
        assert pack.passages
        assert render_book_evidence(pack).startswith("<book_evidence>")
        assert to_prompt_passages(pack)
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# Reranker artifact lifecycle: pinned lock, cache validation, reuse.
# ---------------------------------------------------------------------------


def test_reranker_lock_artifact_is_pinned() -> None:
    path = ROOT / "corpus" / "reranker.lock.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_id"] == RERANKER_MODEL_ID == "BAAI/bge-reranker-v2-m3"
    revision = payload["revision"]
    assert isinstance(revision, str) and len(revision) == 40
    assert all(char in "0123456789abcdef" for char in revision)
    assert payload["sha"] == revision
    assert len(payload["required_files"]) >= 5
    assert "model.safetensors" in payload["required_files"]
    runtime = payload["runtime"]
    assert runtime["interface"] == "FlagEmbedding.FlagReranker"
    assert runtime["device"] == "cpu"
    assert runtime["flag_embedding"]
    lock = load_reranker_lock(path)
    assert lock["model_id"] == RERANKER_MODEL_ID
    source = (ROOT / "src" / "aa" / "retrieval" / "reranker.py").read_text(encoding="utf-8")
    assert "FlagReranker" in source
    assert RERANKER_MODEL_ID in source
    assert "HF_HUB_OFFLINE" in source


def test_reranker_cache_validated_and_reused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aa.retrieval.reranker import (
        reranker_cache_key,
        resolve_hf_cache_dir,
        resolve_reranker_root,
    )

    lock = _reranker_lock()
    assert verify_cached_reranker(tmp_path / "empty", lock) is False
    key_linux = reranker_cache_key(lock, os_name="Linux", arch="X64")
    key_mac = reranker_cache_key(lock, os_name="macOS", arch="ARM64")
    assert key_linux != key_mac
    assert lock["revision"] in key_linux
    # An isolated cache dir holds no snapshot: offline backend, no network.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hfcache"))
    assert resolve_hf_cache_dir() == tmp_path / "hfcache"
    assert resolve_reranker_root(tmp_path / "hfcache").name.startswith("models--BAAI--")
    reset_reranker_cache()
    try:
        import pytest as _pytest

        from aa.retrieval.reranker import RerankerError

        # Production is BGE-only: missing snapshot fails closed.
        with _pytest.raises(RerankerError):
            get_reranker()
        # Explicit hermetic injection only: offline backend on demand.
        first = get_reranker(allow_offline=True)
        second = get_reranker(allow_offline=True)
        assert first is second
        assert first.model_id == RERANKER_MODEL_ID
        assert first.backend == OFFLINE_BACKEND_NAME
        assert len(first.score("тяга вечером", ["тяга к алкоголю мучает"])) == 1
    finally:
        reset_reranker_cache()


def test_flag_backend_scores_when_snapshot_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.retrieval.reranker import resolve_hf_cache_dir, resolve_reranker_root

    lock = _reranker_lock()
    if not verify_cached_reranker(resolve_reranker_root(resolve_hf_cache_dir()), lock):
        pytest.skip("pinned BGE snapshot is not cached; bootstrap first")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    reset_reranker_cache()
    try:
        reranker = get_reranker()
        assert reranker.backend == "bge-reranker-v2-m3-flag/1"
        scores = reranker.score(
            "какая молитва утром помогает?",
            [
                "Молитва утром и медитация вечером помогают многим.",
                "Содружество растет и приглашает новичков.",
            ],
        )
        assert len(scores) == 2
        assert all(isinstance(value, float) for value in scores)
        assert scores[0] > scores[1]
    finally:
        reset_reranker_cache()


# ---------------------------------------------------------------------------
# Frozen benchmark: RRF-only baseline vs target pipeline (offline backend).
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


def test_benchmark_baseline_vs_target_no_regression(tmp_path: pathlib.Path) -> None:
    from aa.qualification.aa_retrieval import RECALL_GATE_AT_5, load_gold
    from aa.qualification.v2_retrieval import (
        duplicate_query_rate,
        planner_proxy_queries,
        query_diversity,
        run_v2_case,
        summarize_v2,
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
        reranker = _offline_reranker()
        results = [run_v2_case(index, case, reranker=reranker) for case in cases]
        summary = summarize_v2(results)
        # No regression vs the RRF-only baseline on the same pool.
        assert summary.target_recall_at_5 >= summary.baseline_recall_at_5
        assert summary.target_recall_at_10 >= summary.baseline_recall_at_10
        # No regression in the existing qualified retrieval recall gate.
        assert summary.target_recall_at_5 >= RECALL_GATE_AT_5
        assert summary.broad_non_regressive is True
        assert summary.duplicate_rate <= 0.05
        assert summary.mean_target_tokens <= RETRIEVED_PASSAGES_BUDGET_TOKENS
        assert summary.p95_target_latency_ms >= summary.p50_target_latency_ms >= 0.0
        assert summary.unsupported_clean_rate >= 0.0
        # Deterministic: same inputs produce the same pack sections.
        repeat = run_v2_case(index, cases[0], reranker=reranker)
        assert repeat.target_sections_10 == results[0].target_sections_10
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
        reranker = _offline_reranker()
        for case in cases:
            result = run_v2_case(index, case, reranker=reranker)
            # Pack winners come from the pool, pool candidates from the union.
            if result.pack_recall:
                assert result.pool_recall
            if result.pool_recall:
                assert result.union_recall
            assert result.reranker_false_negative_loss == (
                result.pool_recall and not result.pack_recall
            )
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# Focused real-reranker lift: frozen hard cases, skipped without the model.
# ---------------------------------------------------------------------------

FOCUSED_LIFT_CASE_IDS = (
    "EN-AA-001",
    "EN-AA-002",
    "EN-AA-005",
    "EN-AA-006",
    "RU-AA-001",
    "RU-AA-002",
)


def test_focused_real_reranker_lift(tmp_path: pathlib.Path) -> None:
    """Spot-check measurable reranker benefit on frozen hard cases.

    Requires the pinned BGE snapshot (bootstrap via
    ``scripts/prefetch_reranker.py``); skipped cleanly when uncached so
    hermetic CI stays fast. The full 52-case validation with the real
    backend is recorded in the benchmark module evaluation.
    """
    from aa.qualification.aa_retrieval import load_gold
    from aa.qualification.v2_retrieval import run_v2_case
    from aa.retrieval.reranker import resolve_hf_cache_dir, resolve_reranker_root

    lock = _reranker_lock()
    if not verify_cached_reranker(resolve_reranker_root(resolve_hf_cache_dir()), lock):
        pytest.skip("pinned BGE snapshot is not cached; bootstrap first")
    index = _gold_index(tmp_path)
    try:
        by_id = {
            case.case_id: case
            for case in load_gold(ROOT / "qualification" / "aa-retrieval.gold.v1.json")
        }
        focused = [by_id[case_id] for case_id in FOCUSED_LIFT_CASE_IDS]
        reset_reranker_cache()
        try:
            reranker = get_reranker()
            assert reranker.model_id == RERANKER_MODEL_ID
            lifts = [run_v2_case(index, case, reranker=reranker) for case in focused]
        finally:
            reset_reranker_cache()
        for result in lifts:
            assert result.target_recall_at_5 >= result.baseline_recall_at_5
        mean_lift = sum(item.reranker_lift_mrr for item in lifts) / len(lifts)
        assert mean_lift > 0, f"focused reranker MRR lift must be positive: {mean_lift}"
    finally:
        close_hybrid_index(index)


# ---------------------------------------------------------------------------
# Graph wiring: planner queries to Evidence Pack state.
# ---------------------------------------------------------------------------


async def test_graph_retrieval_node_wires_evidence_pack(tmp_path: pathlib.Path) -> None:
    from langchain_core.runnables import RunnableLambda

    from aa.conversation.graph import build_turn_graph

    index = _build_index(tmp_path)
    try:
        plan = {"queries": _twelve_queries("срыв и тяга")}

        async def _plan(_messages: Any) -> Any:
            return plan

        offline = _offline_reranker()
        graph = build_turn_graph(
            planner_model=RunnableLambda(_plan),
            retrieval_index=index,
            reranker=offline,
            # Hermetic offline test acceptance only: the frozen BGE
            # validation still exceeds the 5s interactive budget, so
            # production cutover stays blocked without real acceptance.
            performance_accepted=True,
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


def test_v2_retrieval_binding_requires_performance_acceptance(
    tmp_path: pathlib.Path,
) -> None:
    """Binding the slow BGE path fails closed without explicit acceptance."""
    from langchain_core.runnables import RunnableLambda

    from aa.conversation.graph import build_turn_graph

    index = _build_index(tmp_path)
    try:

        async def _plan(_messages: Any) -> Any:
            return {"queries": []}

        offline = _offline_reranker()
        with pytest.raises(ValueError, match="cutover is blocked"):
            build_turn_graph(
                planner_model=RunnableLambda(_plan),
                retrieval_index=index,
                reranker=offline,
            )
        with pytest.raises(ValueError, match="cutover is blocked"):
            build_turn_graph(
                planner_model=RunnableLambda(_plan),
                retrieval_index=index,
                reranker=offline,
                performance_accepted=False,
            )
        # Explicit acceptance wires the node (hermetic offline backend).
        graph = build_turn_graph(
            planner_model=RunnableLambda(_plan),
            retrieval_index=index,
            reranker=offline,
            performance_accepted=True,
        )
        assert graph is not None
    finally:
        close_hybrid_index(index)


def test_make_retrieval_node_requires_performance_acceptance(
    tmp_path: pathlib.Path,
) -> None:
    """Direct node construction fails closed without explicit acceptance."""
    from aa.conversation.retrieval_node import make_retrieval_node

    index = _build_index(tmp_path)
    try:
        offline = _offline_reranker()
        with pytest.raises(ValueError, match="cutover is blocked"):
            make_retrieval_node(index=index, reranker=offline)
        node = make_retrieval_node(index=index, reranker=offline, performance_accepted=True)
        assert callable(node)
    finally:
        close_hybrid_index(index)


async def test_retrieval_node_propagates_latency_budget_signal(
    tmp_path: pathlib.Path,
) -> None:
    """Over-budget BGE turns stay visible in state instead of looking interactive."""
    from aa.conversation.graph import turn_input
    from aa.conversation.retrieval_node import retrieval_node
    from aa.retrieval.evidence import INTERACTIVE_LATENCY_BUDGET_MS

    index = _build_index(tmp_path)
    try:
        state = turn_input("тяга вечером")
        state["search_queries"] = _twelve_queries("тяга поддержка")
        result = await retrieval_node(state, index=index, reranker=_offline_reranker())
        assert result["evidence_pack"]
        latency_ms = result["retrieval_latency_ms"]
        over_budget = result["retrieval_over_budget"]
        assert isinstance(latency_ms, float) and latency_ms >= 0.0
        assert over_budget == (latency_ms > INTERACTIVE_LATENCY_BUDGET_MS)
        # Hermetic offline rerank is fast: it must not read as over budget.
        assert over_budget is False
        skipped = turn_input("привет")
        skipped_result = await retrieval_node(skipped, index=index, reranker=_offline_reranker())
        assert skipped_result["retrieval_latency_ms"] == 0.0
        assert skipped_result["retrieval_over_budget"] is False
    finally:
        close_hybrid_index(index)


def test_frozen_bge_validation_latency_gate_stays_blocked() -> None:
    """The committed 52-case BGE validation fails the 5s gate: cutover stays blocked."""
    from aa.qualification.v2_retrieval import (
        V2Summary,
        require_v2_cutover_acceptance,
        v2_production_status,
        v2_quality_gate,
    )
    from aa.retrieval.evidence import INTERACTIVE_LATENCY_BUDGET_MS

    payload = json.loads(
        (ROOT / "qualification" / "v2-retrieval.bge-validation.v1.json").read_text(encoding="utf-8")
    )
    assert payload["quality_gate"]["passed"] is False
    assert payload["summary"]["cases"] == 52
    assert payload["summary"]["p95_target_latency_ms"] > INTERACTIVE_LATENCY_BUDGET_MS
    assert payload["summary"]["p50_target_latency_ms"] > INTERACTIVE_LATENCY_BUDGET_MS
    assert INTERACTIVE_LATENCY_BUDGET_MS == 5000.0
    summary = V2Summary(
        benchmark_version=str(payload["benchmark_version"]),
        eval_set_version=str(payload["eval_set_version"]),
        cases=int(payload["summary"]["cases"]),
        baseline_recall_at_5=float(payload["summary"]["baseline_recall_at_5"]),
        target_recall_at_5=float(payload["summary"]["target_recall_at_5"]),
        baseline_recall_at_10=float(payload["summary"]["baseline_recall_at_10"]),
        target_recall_at_10=float(payload["summary"]["target_recall_at_10"]),
        baseline_mrr=float(payload["summary"]["baseline_mrr"]),
        target_mrr=float(payload["summary"]["target_mrr"]),
        baseline_ndcg_10=float(payload["summary"]["baseline_ndcg_10"]),
        target_ndcg_10=float(payload["summary"]["target_ndcg_10"]),
        target_coverage=float(payload["summary"]["target_coverage"]),
        duplicate_rate=float(payload["summary"]["duplicate_rate"]),
        mean_reranker_lift_recall=float(payload["summary"]["mean_reranker_lift_recall"]),
        mean_reranker_lift_mrr=float(payload["summary"]["mean_reranker_lift_mrr"]),
        mean_reranker_lift_ndcg=float(payload["summary"]["mean_reranker_lift_ndcg"]),
        mean_query_diversity=float(payload["summary"]["mean_query_diversity"]),
        mean_duplicate_query_rate=float(payload["summary"]["mean_duplicate_query_rate"]),
        mean_target_tokens=float(payload["summary"]["mean_target_tokens"]),
        p50_target_latency_ms=float(payload["summary"]["p50_target_latency_ms"]),
        p95_target_latency_ms=float(payload["summary"]["p95_target_latency_ms"]),
        unsupported_clean_rate=float(payload["summary"]["unsupported_clean_rate"]),
        broad_non_regressive=bool(payload["summary"]["broad_non_regressive"]),
        pool_to_pack_loss_rate=float(payload["summary"]["pool_to_pack_loss_rate"]),
        peak_rss_mb=0.0,
    )
    passed, failures = v2_quality_gate(summary)
    assert passed is False
    assert any("p95 latency" in failure for failure in failures)
    status = v2_production_status(summary)
    assert status["quality_gate_passed"] is False
    assert status["cutover_allowed"] is False
    assert status["production_promotion_blocked"] is True
    with pytest.raises(ValueError, match="cutover is blocked"):
        require_v2_cutover_acceptance(performance_accepted=False)
