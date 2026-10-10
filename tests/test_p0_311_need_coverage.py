"""AUDIT P1 (kodmial/aa#311): per-query and per-need search exposure.

Regressions on invented fixture text (no canonical book text):

- two-part request with many high-ranking near-duplicates for need A and
  one low-fused-rank candidate for need B: both appear in the selector's
  actual prompt and can be selected;
- paraphrase queries for need A do not earn unbounded priority over need B;
- one candidate satisfying A and B dedups text but keeps both associations;
- no genuine evidence for B reports uncovered / requests more search and
  never fabricates coverage from irrelevant chapters;
- deep-rank (>64) candidate for B is discoverable within the bounded
  selector-input budget and the #303 rank-66 follow-up stays reachable;
- actual prompt/structured selection outputs are asserted, not only the
  internal fused pool;
- old full pack plus a genuinely relevant new passage with lower lexical
  overlap (second need) survives the combined-pack budget boundary via
  coverage-aware budgeting, not lexical order alone.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

from aa.conversation.conversation_context import (
    InformationNeed,
    build_query_need_map,
    needs_from_plan,
)
from aa.conversation.semantic_selection import (
    MAX_SELECTION_CANDIDATES,
    assess_need_preview_coverage,
    select_need_aware_previews,
    selection_prompt,
    selection_telemetry,
)
from aa.conversation.turn_pipeline import MAX_PACK_PASSAGES, merge_pack_dicts
from aa.retrieval.evidence import candidate_need_provenance, candidate_query_provenance
from aa.retrieval.fusion import FusedCandidate


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _needs_two() -> list[InformationNeed]:
    return [
        InformationNeed(need_id="need-1", text="утренний разбор тяги"),
        InformationNeed(need_id="need-2", text="вечерняя поддержка сообщества"),
    ]


def _fused(chunk_id: str, score: float) -> FusedCandidate:
    return FusedCandidate(
        chunk_id=chunk_id,
        fused_score=score,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=score,
        dense_score=score,
    )


def test_query_need_map_is_structural_not_lexical() -> None:
    queries = ["утренний разбор тяги", "вечерняя поддержка сообщества"]
    needs = _needs_two()
    # Structural derivation maps each query to its identical need text.
    mapped = build_query_need_map(queries, needs)
    assert [m.query_id for m in mapped] == ["q1", "q2"]
    assert mapped[0].need_ids == ["need-1"]
    assert mapped[1].need_ids == ["need-2"]
    # Planner-emitted links with an unknown id never fabricate coverage.
    linked = build_query_need_map(queries, needs, planner_links=[["need-1"], ["need-99"]])
    assert linked[0].need_ids == ["need-1"]
    assert linked[1].need_ids == []
    # Needs that do not match the plan stay unmapped (unknown/unmapped).
    other = [InformationNeed(need_id="need-1", text="совсем другой смысл")]
    unmapped = build_query_need_map(queries, other)
    assert unmapped[0].need_ids == []
    assert unmapped[1].need_ids == []
    # Stable through replay.
    again = build_query_need_map(queries, needs)
    assert [m.need_ids for m in again] == [m.need_ids for m in mapped]


def test_candidate_provenance_keeps_many_to_many() -> None:
    per_query_ids = [["c-a1", "c-shared"], ["c-b1", "c-shared"]]
    query_map = candidate_query_provenance(per_query_ids)
    assert query_map["c-shared"] == ["q1", "q2"]
    assert query_map["c-a1"] == ["q1"]
    needs = _needs_two()
    queries = ["утренний разбор тяги", "вечерняя поддержка сообщества"]
    qmap = build_query_need_map(queries, needs)
    qmap_dicts = [{"query_id": m.query_id, "need_ids": list(m.need_ids)} for m in qmap]
    need_map = candidate_need_provenance(query_map, qmap_dicts)
    assert sorted(need_map["c-shared"]) == ["need-1", "need-2"]
    assert need_map["c-a1"] == ["need-1"]
    assert need_map["c-b1"] == ["need-2"]


def test_two_need_low_rank_candidate_reaches_actual_prompt() -> None:
    needs = _needs_two()
    queries = ["утренний разбор тяги", "вечерняя поддержка сообщества"]
    qmap = build_query_need_map(queries, needs)
    qmap_dicts = [{"query_id": m.query_id, "need_ids": list(m.need_ids)} for m in qmap]
    # Many high-ranking near-duplicates for need A, one low-rank for need B.
    fused_ordered: list[tuple[str, float]] = []
    texts: dict[str, str] = {}
    per_query_a: list[str] = []
    for pos in range(30):
        cid = f"chapter-1:ru:a{pos:04d}"
        fused_ordered.append((cid, float(100 - pos)))
        texts[cid] = f"Утренний разбор тяги вариант {pos}."
        per_query_a.append(cid)
    low_cid = "chapter-7:ru:b0001"
    fused_ordered.append((low_cid, 1.0))
    texts[low_cid] = "Вечерняя поддержка сообщества рядом."
    per_query_ids = [per_query_a, [low_cid]]
    query_map = candidate_query_provenance(per_query_ids)
    need_map = candidate_need_provenance(query_map, qmap_dicts)
    previews = select_need_aware_previews(
        fused_ordered=fused_ordered,
        texts=texts,
        sections={cid: "chapter-1" for cid, _ in fused_ordered},
        sources={cid: "ru-fourth-edition-txt" for cid, _ in fused_ordered},
        candidate_query_map=query_map,
        candidate_need_map=need_map,
        information_needs=needs,
        limit=MAX_SELECTION_CANDIDATES,
    )
    ids = {p.chunk_id for p in previews}
    assert low_cid in ids
    assert any(p.chunk_id.startswith("chapter-1:ru:a") for p in previews)
    # Both needs appear in the selector's actual prompt, not only the pool.
    _, prompt = selection_prompt(
        previews=previews,
        resolved_intent="утренний разбор и вечерняя поддержка",
        information_needs=needs,
    )
    assert low_cid in prompt
    assert "need-1" in prompt and "need-2" in prompt
    assert "вечерняя поддержка сообщества" in prompt
    statuses, _ = assess_need_preview_coverage(previews, needs)
    assert all(s.represented for s in statuses)


def test_paraphrase_queries_do_not_starve_independent_need() -> None:
    needs = _needs_two()
    # Paraphrase queries for need A must not starve independent need B.
    deduped = ["утренний разбор тяги", "вечерняя поддержка сообщества"]
    qmap = build_query_need_map(deduped, needs)
    assert len(qmap) == 2
    qmap_dicts = [{"query_id": m.query_id, "need_ids": list(m.need_ids)} for m in qmap]
    assert qmap_dicts[0]["need_ids"] == ["need-1"]
    assert qmap_dicts[1]["need_ids"] == ["need-2"]
    fused_ordered = [(f"chapter-1:ru:a{i:04d}", float(50 - i)) for i in range(20)]
    fused_ordered.append(("chapter-7:ru:b0001", 1.0))
    texts = {cid: f"Текст {cid}." for cid, _ in fused_ordered}
    per_query_ids = [[cid for cid, _ in fused_ordered[:20]], ["chapter-7:ru:b0001"]]
    query_map = candidate_query_provenance(per_query_ids)
    need_map = candidate_need_provenance(query_map, qmap_dicts)
    previews = select_need_aware_previews(
        fused_ordered=fused_ordered,
        texts=texts,
        candidate_query_map=query_map,
        candidate_need_map=need_map,
        information_needs=needs,
        limit=8,
    )
    assert "chapter-7:ru:b0001" in {p.chunk_id for p in previews}
    # No unbounded priority: need B keeps representation within a tiny budget.
    counts = {"need-1": 0, "need-2": 0}
    for preview in previews:
        for nid in preview.need_ids:
            counts[nid] = counts.get(nid, 0) + 1
    assert counts["need-2"] >= 1
    assert counts["need-1"] <= 7


def test_shared_candidate_dedups_without_losing_needs() -> None:
    needs = _needs_two()
    queries = ["утренний разбор тяги", "вечерняя поддержка сообщества"]
    qmap = build_query_need_map(queries, needs)
    qmap_dicts = [{"query_id": m.query_id, "need_ids": list(m.need_ids)} for m in qmap]
    per_query_ids = [["c-shared", "c-a1"], ["c-shared", "c-b1"]]
    query_map = candidate_query_provenance(per_query_ids)
    need_map = candidate_need_provenance(query_map, qmap_dicts)
    assert sorted(need_map["c-shared"]) == ["need-1", "need-2"]
    fused_ordered = [("c-shared", 10.0), ("c-a1", 9.0), ("c-b1", 8.0)]
    texts = {cid: f"Текст {cid}." for cid, _ in fused_ordered}
    previews = select_need_aware_previews(
        fused_ordered=fused_ordered,
        texts=texts,
        candidate_query_map=query_map,
        candidate_need_map=need_map,
        information_needs=needs,
        limit=8,
    )
    shared = [p for p in previews if p.chunk_id == "c-shared"]
    assert len(shared) == 1
    assert sorted(shared[0].need_ids) == ["need-1", "need-2"]


async def test_no_evidence_for_second_need_reports_uncovered() -> None:
    from aa.conversation.semantic_selection import aselect_semantic_candidates

    needs = _needs_two()
    queries = ["утренний разбор тяги", "вечерняя поддержка сообщества"]
    qmap = build_query_need_map(queries, needs)
    qmap_dicts = [{"query_id": m.query_id, "need_ids": list(m.need_ids)} for m in qmap]
    # Only need-A evidence exists; nothing genuine for need-2.
    fused_ordered = [(f"chapter-1:ru:a{i:04d}", float(10 - i)) for i in range(5)]
    texts = {cid: f"Утренний разбор тяги {cid}." for cid, _ in fused_ordered}
    per_query_ids = [[cid for cid, _ in fused_ordered], []]
    query_map = candidate_query_provenance(per_query_ids)
    need_map = candidate_need_provenance(query_map, qmap_dicts)
    previews = select_need_aware_previews(
        fused_ordered=fused_ordered,
        texts=texts,
        candidate_query_map=query_map,
        candidate_need_map=need_map,
        information_needs=needs,
        limit=MAX_SELECTION_CANDIDATES,
    )
    statuses, _ = assess_need_preview_coverage(previews, needs)
    by_id = {s.need_id: s for s in statuses}
    assert by_id["need-1"].represented is True
    assert by_id["need-2"].represented is False

    first_id = fused_ordered[0][0]

    class _Model:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
        ) -> object:
            _ = (prompt, system, schema, retry_count)
            assert "need-2" in prompt
            assert first_id in prompt
            return {
                "selected_chunk_ids": [first_id],
                "need_more_detail": True,
                "followup_queries": ["вечерняя поддержка сообщества рядом"],
                "uncovered_need_ids": ["need-2"],
            }

    selection = await aselect_semantic_candidates(
        previews,
        resolved_intent="утренний разбор и вечерняя поддержка",
        model=_Model(),
        known_chunk_ids={cid for cid, _ in fused_ordered},
        information_needs=needs,
    )
    assert selection.uncovered_need_ids == ["need-2"]
    assert selection.need_more_detail is True
    # No fabricated coverage: nothing irrelevant is assigned to need-2.
    assert all("need-2" not in p.need_ids for p in previews)
    telemetry = selection_telemetry(previews=previews, selection=selection, information_needs=needs)
    assert telemetry["selection_need_unrepresented"] == ["need-2"]
    assert telemetry["selection_uncovered_need_ids"] == ["need-2"]


async def test_deep_rank_beyond_64_for_second_need_within_budget() -> None:
    needs = _needs_two()
    queries = ["утренний разбор тяги", "вечерняя поддержка сообщества"]
    qmap = build_query_need_map(queries, needs)
    qmap_dicts = [{"query_id": m.query_id, "need_ids": list(m.need_ids)} for m in qmap]
    fused_ordered: list[tuple[str, float]] = []
    texts: dict[str, str] = {}
    per_a: list[str] = []
    for pos in range(70):
        cid = f"chapter-1:ru:a{pos:04d}"
        fused_ordered.append((cid, float(200 - pos)))
        texts[cid] = f"Утренний разбор тяги фон {pos}."
        per_a.append(cid)
    deep_cid = "chapter-7:ru:b0066"
    fused_ordered.append((deep_cid, 0.5))
    texts[deep_cid] = "Вечерняя поддержка сообщества рядом."
    per_query_ids = [per_a, [deep_cid]]
    query_map = candidate_query_provenance(per_query_ids)
    need_map = candidate_need_provenance(query_map, qmap_dicts)
    previews = select_need_aware_previews(
        fused_ordered=fused_ordered,
        texts=texts,
        candidate_query_map=query_map,
        candidate_need_map=need_map,
        information_needs=needs,
        limit=MAX_SELECTION_CANDIDATES,
    )
    assert len(previews) <= MAX_SELECTION_CANDIDATES
    assert deep_cid in {p.chunk_id for p in previews}
    deep = next(p for p in previews if p.chunk_id == deep_cid)
    assert deep.fused_rank == 70
    assert "need-2" in deep.need_ids
    _, prompt = selection_prompt(
        previews=previews,
        resolved_intent="утренний разбор и вечерняя поддержка",
        information_needs=needs,
    )
    assert deep_cid in prompt


def _record(
    chunk_id: str, *, section: str = "chapter-1", start: int = 0, text: str = "Текст."
) -> Any:
    from aa.retrieval.index import ChunkRecord

    end = start + len(text)
    return ChunkRecord(
        chunk_id=chunk_id,
        logical_chunk_id=chunk_id,
        section=section,
        book="aa-big-book",
        parent="p-" + chunk_id,
        prev=None,
        next=None,
        source_id="ru-fourth-edition-txt",
        source_file="corpus/source/raw-ru/aa-big-book.txt",
        source_sha256="s" * 64,
        char_start=start,
        char_end=end,
        text_sha256=_sha(text),
        text=text,
        corpus_version="ru-v1",
    )


def _fake_index(records: list[Any]) -> Any:
    return SimpleNamespace(
        chunks={record.chunk_id: record for record in records},
        metadata={
            "ru_artifact_sha256": "r" * 64,
            "embedding_backend": "hashing",
            "embedding_dim": 64,
        },
        dense=None,
        lexical_conn=None,
        ram_resident=True,
    )


def _pack_dict(passage_id: str, text: str, **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": passage_id.split("#")[0] if "#" in passage_id else "chapter-1",
        "child_chunk_ids": [passage_id],
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": _sha(text),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }
    base.update(extra)
    return base


def test_rank66_followup_stays_reachable_with_needs(monkeypatch: Any) -> None:
    import asyncio

    from aa.conversation import retrieval_node as retrieval_node_mod

    records: list[Any] = []
    for pos in range(70):
        text = f"Фоновый отрывок {pos}."
        if pos == 65:
            text = "Решающий отрывок про трезвый утренний разбор тяги."
        records.append(_record(f"chapter-1:ru:c{pos:04d}", start=pos * 100, text=text))
    index = _fake_index(records)
    first_fused = {
        record.chunk_id: _fused(record.chunk_id, float(70 - pos))
        for pos, record in enumerate(records)
    }
    target = records[65].chunk_id
    second_fused = {
        target: _fused(target, 100.0),
        records[0].chunk_id: _fused(records[0].chunk_id, 1.0),
    }

    def _fake_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        return ([], [[]])

    calls = {"count": 0}

    def _fake_fuse(ranked: Any, per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, per_ids, kwargs)
        calls["count"] += 1
        fused = dict(first_fused) if calls["count"] == 1 else dict(second_fused)
        return fused, list(fused)

    from aa.retrieval import evidence as evidence_mod

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)

    class _EmptyModel:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "selected_chunk_ids": [],
                "need_more_detail": True,
                "followup_queries": ["уточняющий запрос про тягу"],
            }

    needs = needs_from_plan("трезвый утренний разбор тяги", ["первый запрос"])
    pack = asyncio.run(
        retrieval_node_mod.aretrieve_with_semantic_selection(
            index,
            ["первый запрос"],
            resolved_intent="трезвый утренний разбор тяги",
            conversation_context="",
            user_message="разобрать тягу",
            selection_model=_EmptyModel(),
            information_needs=needs,
        )
    )
    assert pack.retrieval_metadata["followup_added"] >= 1
    assert pack.passages
    texts_joined = " ".join(p.exact_text for p in pack.passages)
    assert "Решающий отрывок" in texts_joined


async def test_actual_prompt_selection_covers_both_needs(monkeypatch: Any) -> None:
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection
    from aa.retrieval import evidence as evidence_mod

    records: list[Any] = []
    for pos in range(10):
        records.append(
            _record(
                f"chapter-1:ru:a{pos:04d}",
                section="chapter-1",
                start=pos * 100,
                text=f"Утренний разбор тяги фон {pos}.",
            )
        )
    records.append(
        _record(
            "chapter-7:ru:b0001",
            section="chapter-7",
            start=5000,
            text="Вечерняя поддержка сообщества рядом.",
        )
    )
    index = _fake_index(records)
    fused_pool = {record.chunk_id: _fused(record.chunk_id, 10.0) for record in records}
    fused_pool["chapter-7:ru:b0001"] = _fused("chapter-7:ru:b0001", 9.0)

    def _fake_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, kwargs)
        per_ids: list[list[str]] = []
        for query in queries:
            if "утренн" in query:
                per_ids.append([f"chapter-1:ru:a{pos:04d}" for pos in range(10)])
            else:
                per_ids.append(["chapter-7:ru:b0001"])
        return ([], per_ids)

    def _fake_fuse(ranked: Any, per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, kwargs)
        return dict(fused_pool), list(fused_pool)

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)

    queries = ["утренний разбор тяги", "вечерняя поддержка сообщества"]
    needs = _needs_two()

    seen: dict[str, str] = {}

    class _BothModel:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
        ) -> object:
            _ = (system, schema, retry_count)
            seen["prompt"] = prompt
            assert "chapter-7:ru:b0001" in prompt
            assert "need-1" in prompt and "need-2" in prompt
            return {
                "selected_chunk_ids": ["chapter-1:ru:a0000", "chapter-7:ru:b0001"],
                "need_more_detail": False,
                "followup_queries": [],
            }

    pack = await aretrieve_with_semantic_selection(
        index,
        queries,
        resolved_intent="утренний разбор и вечерняя поддержка",
        conversation_context="",
        user_message="как разбирать утром и что вечером",
        selection_model=_BothModel(),
        information_needs=needs,
    )
    assert pack.passages
    assert "selection_preview_map" in pack.retrieval_metadata
    preview_map = pack.retrieval_metadata["selection_preview_map"]
    by_id = {entry["chunk_id"]: entry for entry in preview_map}
    assert by_id["chapter-7:ru:b0001"]["selected"] is True
    assert "need-2" in by_id["chapter-7:ru:b0001"]["need_ids"]
    assert pack.retrieval_metadata["uncovered_need_ids"] == []
    joined = " ".join(p.exact_text for p in pack.passages)
    assert "Вечерняя поддержка" in joined


def test_merge_budget_boundary_keeps_second_need_despite_lexical_gap() -> None:
    old = [
        _pack_dict(f"chapter-1#old-{index}", f"Неотносящийся текст про погоду номер {index}.")
        for index in range(MAX_PACK_PASSAGES)
    ]
    # Genuinely relevant new passage for the second need with low lexical
    # overlap against the intent prefix (lexical heuristic alone buries it).
    fresh_text = "Вечером рядом сообщество поддерживает спокойно без разбора."
    fresh = [_pack_dict("chapter-7#new-relevant", fresh_text, need_ids=["need-2"])]
    for entry in old:
        entry["need_ids"] = ["need-1"]
    needs = _needs_two()
    merged = merge_pack_dicts(
        old,
        fresh,
        resolved_intent="трезвый утренний разбор тяги поддержка",
        conversation_context="",
        information_needs=[{"need_id": "need-1"}, {"need_id": "need-2"}],
    )
    assert "chapter-7#new-relevant" in {item["passage_id"] for item in merged}
    assert len(merged) <= MAX_PACK_PASSAGES
    # Shared passage unions associations instead of dropping one need.
    dup_old = [_pack_dict("chapter-1#shared", "Общий текст.", need_ids=["need-1"])]
    dup_new = [_pack_dict("chapter-1#shared", "Общий текст.", need_ids=["need-2"])]
    reunited = merge_pack_dicts(dup_old, dup_new)
    assert reunited[0]["need_ids"] == ["need-1", "need-2"]
    _ = needs
