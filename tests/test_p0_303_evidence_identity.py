"""AUDIT P0-1 (kodmial/aa#303): evidence identity and follow-up retrieval.

Focused regressions on invented fixture text (no canonical book text):

- two different ranges from the same chapter across two searches stay distinct;
- rank-66 candidate unseen among 64 previews stays promotable on follow-up;
- empty selection + useful follow-up query is valid and executed;
- a newly relevant source semantically displaces an older irrelevant one
  in an already full pack under budget;
- an actually identical range is deduplicated;
- the same stable id with different bytes fails integrity;
- rate-limit errors propagate through the typed path.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

from aa.conversation import retrieval_node as retrieval_node_mod
from aa.conversation.retrieval_node import pack_to_state
from aa.conversation.semantic_selection import (
    SemanticSelectionError,
    validate_semantic_selection,
)
from aa.conversation.turn_pipeline import MAX_PACK_PASSAGES, merge_pack_dicts
from aa.retrieval.evidence import (
    EvidenceIntegrityError,
    EvidencePack,
    EvidencePassageData,
    expand_small_to_big,
    stable_passage_id,
)
from aa.retrieval.fusion import FusedCandidate
from aa.retrieval.index import ChunkRecord


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    chunk_id: str,
    *,
    section: str = "chapter-1",
    start: int = 0,
    text: str = "Текст.",
    source_sha: str = "s" * 64,
    parent: str = "p",
) -> ChunkRecord:
    end = start + len(text)
    return ChunkRecord(
        chunk_id=chunk_id,
        logical_chunk_id=chunk_id,
        section=section,
        book="aa-big-book",
        parent=parent,
        prev=None,
        next=None,
        source_id="ru-fourth-edition-txt",
        source_file="corpus/source/raw-ru/aa-big-book.txt",
        source_sha256=source_sha,
        char_start=start,
        char_end=end,
        text_sha256=_sha(text),
        text=text,
        corpus_version="ru-v1",
    )


def _fake_index(records: list[ChunkRecord]) -> Any:
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


def _winner(chunk_id: str, score: float) -> FusedCandidate:
    return FusedCandidate(
        chunk_id=chunk_id,
        fused_score=score,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=1.0,
        dense_score=1.0,
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


def test_two_ranges_same_chapter_stay_distinct_and_stable() -> None:
    first = _record("chapter-1:ru:c0001", start=0, text="Первый отрывок про тягу.", parent="p1")
    second = _record("chapter-1:ru:c0002", start=100, text="Второй отрывок про тягу.", parent="p2")
    index = _fake_index([first, second])
    once = expand_small_to_big(index, [_winner(first.chunk_id, 2.0)], neighbor_window=0)
    twice = expand_small_to_big(index, [_winner(second.chunk_id, 1.0)], neighbor_window=0)
    assert len(once) == 1 and len(twice) == 1
    assert once[0].passage_id != twice[0].passage_id
    repeat = expand_small_to_big(index, [_winner(first.chunk_id, 2.0)], neighbor_window=0)
    assert repeat[0].passage_id == once[0].passage_id
    assert "#" in once[0].passage_id and "exp0000" not in once[0].passage_id
    assert "atom-" not in twice[0].passage_id


def test_identical_range_deduplicated() -> None:
    text = "Тот же самый отрывок про поддержку."
    entry = _pack_dict("chapter-1#abc:0-10:deadbeef", text)
    merged = merge_pack_dicts([entry], [dict(entry)])
    assert len(merged) == 1
    assert merged[0]["passage_id"] == entry["passage_id"]


def test_same_stable_id_different_content_fails_integrity() -> None:
    first = _pack_dict("stable-id-1", "Первый текст про тягу.")
    second = _pack_dict("stable-id-1", "Совсем другой текст про тягу.")
    with pytest.raises(EvidenceIntegrityError):
        merge_pack_dicts([first], [second])


def test_source_provenance_survives_pack_to_state() -> None:
    text = "Точный текст про утренний разбор."
    passage = EvidencePassageData(
        passage_id=stable_passage_id(
            source_sha256="s" * 64,
            section_id="chapter-3",
            char_start=0,
            char_end=len(text),
            text_sha256=_sha(text),
        ),
        exact_text=text,
        source_id="ru-fourth-edition-txt",
        section_id="chapter-3",
        child_chunk_ids=("chapter-3:ru:c0001",),
        char_start=0,
        char_end=len(text),
        text_sha256=_sha(text),
        source_sha256="s" * 64,
    )
    pack = EvidencePack(
        passages=(passage,),
        total_tokens=10,
        corpus_version="r" * 64,
        retrieval_metadata={},
    )
    _, dicts = pack_to_state(pack)
    assert dicts[0]["source_sha256"] == "s" * 64
    assert dicts[0]["corpus_version"] == "r" * 64
    assert dicts[0]["text_sha256"] == _sha(text)


def test_empty_selection_with_followup_is_valid() -> None:
    selection = validate_semantic_selection(
        {
            "selected_chunk_ids": [],
            "need_more_detail": True,
            "followup_queries": ["утренний разбор тяги"],
        },
        known_chunk_ids=set(),
    )
    assert selection.selected_chunk_ids == []
    assert selection.need_more_detail is True
    assert selection.followup_queries == ["утренний разбор тяги"]


def test_empty_selection_without_justification_rejected() -> None:
    with pytest.raises(SemanticSelectionError):
        validate_semantic_selection(
            {
                "selected_chunk_ids": [],
                "need_more_detail": False,
                "followup_queries": [],
            },
            known_chunk_ids=set(),
        )


def test_full_pack_displaces_irrelevant_with_relevant() -> None:
    intent = "трезвый утренний разбор тяги поддержка"
    old = [
        _pack_dict(f"old-{index}", f"Неотносящийся текст про погоду номер {index}.")
        for index in range(MAX_PACK_PASSAGES)
    ]
    fresh_text = "Трезвый утренний разбор тяги и поддержка рядом."
    fresh = [_pack_dict("new-relevant", fresh_text)]
    merged = merge_pack_dicts(old, fresh, resolved_intent=intent, conversation_context="")
    ids = {item["passage_id"] for item in merged}
    assert "new-relevant" in ids
    assert len(merged) <= MAX_PACK_PASSAGES
    assert len(merged) == MAX_PACK_PASSAGES


class _EmptyModel:
    def __init__(self, followups: list[str]) -> None:
        self._followups = followups

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return {
            "selected_chunk_ids": [],
            "need_more_detail": True,
            "followup_queries": list(self._followups),
        }


def _fused(chunk_id: str, score: float) -> FusedCandidate:
    return FusedCandidate(
        chunk_id=chunk_id,
        fused_score=score,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=score,
        dense_score=score,
    )


async def _run_rank66_promotion(monkeypatch: Any) -> EvidencePack:
    records: list[ChunkRecord] = []
    for pos in range(70):
        text = f"Фоновый отрывок {pos}."
        if pos == 65:
            text = "Решающий отрывок про трезвый утренний разбор тяги."
        records.append(
            _record(
                f"chapter-1:ru:c{pos:04d}",
                start=pos * 100,
                text=text,
                parent=f"p{pos}",
            )
        )
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
        if calls["count"] == 1:
            fused = dict(first_fused)
        else:
            fused = dict(second_fused)
        pool = list(fused)
        return fused, pool

    from aa.retrieval import evidence as evidence_mod

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)
    pack = await retrieval_node_mod.aretrieve_with_semantic_selection(
        index,
        ["первый запрос"],
        resolved_intent="трезвый утренний разбор тяги",
        conversation_context="",
        user_message="разобрать тягу",
        selection_model=_EmptyModel(["уточняющий запрос про тягу"]),
    )
    return pack


def test_rank66_unseen_candidate_promoted_on_followup(monkeypatch: Any) -> None:
    import asyncio

    pack = asyncio.run(_run_rank66_promotion(monkeypatch))
    assert pack.retrieval_metadata["followup_added"] >= 1
    assert pack.passages, "follow-up promotion must yield evidence"
    discovered = pack.retrieval_metadata["discovered_ids"]
    previewed = pack.retrieval_metadata["previewed_ids"]
    read = pack.retrieval_metadata["read_ids"]
    assert len(previewed) == 64
    assert len(discovered) >= 64
    assert set(previewed) < set(discovered) or len(discovered) > len(previewed)
    assert read, "read ids must be observable"
    texts = " ".join(passage.exact_text for passage in pack.passages)
    assert "Решающий отрывок" in texts


async def _run_explicit_empty_no_fallback(monkeypatch: Any) -> EvidencePack:
    record = _record("chapter-1:ru:c0001", start=0, text="Фоновый текст без пользы.")
    index = _fake_index([record])
    fused = {record.chunk_id: _fused(record.chunk_id, 5.0)}

    def _fake_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        return ([], [[]])

    def _fake_fuse(ranked: Any, per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, per_ids, kwargs)
        return dict(fused), [record.chunk_id]

    from aa.retrieval import evidence as evidence_mod

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)
    return await retrieval_node_mod.aretrieve_with_semantic_selection(
        index,
        ["первый запрос"],
        resolved_intent="трезвый утренний разбор тяги",
        conversation_context="",
        user_message="разобрать тягу",
        selection_model=_EmptyModel(["тот же самый запрос"]),
    )


def test_explicit_empty_followup_without_fresh_stays_empty(monkeypatch: Any) -> None:
    import asyncio

    pack = asyncio.run(_run_explicit_empty_no_fallback(monkeypatch))
    assert pack.passages == ()


def test_followup_rate_limit_propagates_typed(monkeypatch: Any) -> None:
    import asyncio

    from aa.opencode.errors import OpenCodeRateLimitError

    record = _record("chapter-1:ru:c0001", start=0, text="Фоновый текст.")
    index = _fake_index([record])
    calls = {"count": 0}

    def _flaky_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        calls["count"] += 1
        if calls["count"] >= 2:
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")
        return ([], [[]])

    def _fake_fuse(ranked: Any, per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, per_ids, kwargs)
        return {record.chunk_id: _fused(record.chunk_id, 5.0)}, [record.chunk_id]

    from aa.retrieval import evidence as evidence_mod

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _flaky_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)
    with pytest.raises(OpenCodeRateLimitError):
        asyncio.run(
            retrieval_node_mod.aretrieve_with_semantic_selection(
                index,
                ["первый запрос"],
                resolved_intent="трезвый утренний разбор тяги",
                conversation_context="",
                user_message="разобрать тягу",
                selection_model=_EmptyModel(["уточняющий запрос"]),
            )
        )
