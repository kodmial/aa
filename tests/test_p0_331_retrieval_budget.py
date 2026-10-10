"""P0 bounded retrieval budget (kodmial/aa#331).

Single architectural correction: the shared read/coverage loop must not
grind repeated sequential selector/coverage provider calls. Redundant
work stops on stable preview identity (same candidates/ranks/need seam),
canonical reads reuse a validated cache, and every model/retriever await
is bounded by the remaining interactive deadline with typed
``exhausted/latency-budget`` (never fake coverage, never a nested retry).
A new canonical range, material expansion, or newly covered need is
progress and continues; the same unsuccessful fingerprint cannot loop.
Provider 429 always propagates.
"""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    chunk_id: str,
    *,
    section: str = "chapter-1",
    start: int = 0,
    text: str = "Текст про поддержку.",
    parent: str | None = None,
) -> Any:
    from aa.retrieval.index import ChunkRecord

    end = start + len(text)
    return ChunkRecord(
        chunk_id=chunk_id,
        logical_chunk_id=chunk_id,
        section=section,
        book="aa-big-book",
        parent=(parent or ("p-" + chunk_id)),
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


def _fused(chunk_id: str, score: float) -> Any:
    from aa.retrieval.fusion import FusedCandidate

    return FusedCandidate(
        chunk_id=chunk_id,
        fused_score=score,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=score,
        dense_score=score,
    )


class _ScriptedModel:
    """Serve scripted selections; coverage judges marker presence in passages."""

    def __init__(
        self,
        selections: list[dict[str, Any]],
        markers: dict[str, str],
        *,
        delay_s: float = 0.0,
    ) -> None:
        self._selections = list(selections)
        self._markers = dict(markers)
        self._delay = float(delay_s)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        import re as _re

        _ = (system, schema, retry_count)
        self.calls += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if "<passages>" in prompt and "<original_request>" in prompt:
            passage_texts: dict[str, str] = {}
            for match in _re.finditer(
                r'<passage id="([^"]+)"[^>]*>(.*?)</passage>', prompt, _re.DOTALL
            ):
                pid, body = match.group(1), match.group(2)
                passage_texts[pid] = " ".join(body.split())
            needs: list[dict[str, object]] = []
            for nid, marker in self._markers.items():
                supporting = [pid for pid, body in passage_texts.items() if marker in body]
                if supporting:
                    needs.append(
                        {
                            "need_id": nid,
                            "covered": True,
                            "supporting_passage_ids": supporting,
                            "supporting_quote": marker,
                            "missing": "",
                        }
                    )
                else:
                    needs.append(
                        {
                            "need_id": nid,
                            "covered": False,
                            "supporting_passage_ids": [],
                            "supporting_quote": "",
                            "missing": f"no passage states {marker}",
                        }
                    )
            return {"needs": needs}
        if self._selections:
            return dict(self._selections.pop(0))
        return {
            "selected_chunk_ids": [],
            "need_more_detail": False,
            "followup_queries": [],
        }


def _install_branch(
    monkeypatch: Any, pools: list[dict[str, Any]], per_ids: list[list[str]]
) -> dict[str, Any]:
    import aa.retrieval.evidence as evidence_mod

    calls = {"count": 0}

    def _fake_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        return ([], [list(ids) for ids in per_ids])

    def _fake_fuse(ranked: Any, _per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, _per_ids, kwargs)
        pos = min(calls["count"], len(pools) - 1)
        calls["count"] += 1
        fused = dict(pools[pos])
        return fused, list(fused)

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)
    return calls


@pytest.mark.parametrize(
    "intent",
    [
        "как разбирать утреннюю тягу и вечернее беспокойство",
        "утром тянет, а к вечеру тревожно — как быть",
        "разбор тяги с утра плюс вечерняя тревога",
        "что делать с утренней тягой и вечерним беспокойством",
    ],
)
async def test_two_need_compound_completes_with_real_evidence(
    monkeypatch: Any, intent: str
) -> None:
    """Two-need compound question closes on genuine evidence for paraphrases."""
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Утром помогает разбор АЛЬФА-МАРКЕР рядом.")
    second = _record("chapter-2:ru:second", start=0, text="Вечером помогает шаг БЕТА-МАРКЕР рядом.")
    index = _fake_index([first, second])
    _install_branch(
        monkeypatch,
        [
            {
                first.chunk_id: _fused(first.chunk_id, 50.0),
                second.chunk_id: _fused(second.chunk_id, 49.0),
            }
        ],
        [[first.chunk_id, second.chunk_id]],
    )
    model = _ScriptedModel(
        selections=[
            {
                "selected_chunk_ids": [first.chunk_id, second.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            }
        ],
        markers={"need-1": "АЛЬФА-МАРКЕР", "need-2": "БЕТА-МАРКЕР"},
    )
    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор тяги", "вечернее беспокойство шаги"],
        resolved_intent=intent,
        conversation_context="",
        user_message=intent,
        selection_model=model,
        information_needs=[
            {"need_id": "need-1", "text": "утренний разбор тяги"},
            {"need_id": "need-2", "text": "вечернее беспокойство"},
        ],
        max_iterations=3,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "ready"
    assert metadata.get("coverage_all_covered") is True
    assert set(metadata.get("coverage_missing_need_ids", [])) == set()
    # Genuine evidence: both markers present in served exact text (no
    # found==covered shortcut: coverage needed cited passages).
    served = " ".join(str(p.exact_text) for p in pack.passages)
    assert "АЛЬФА-МАРКЕР" in served and "БЕТА-МАРКЕР" in served
    assert len(pack.passages) >= 1


async def test_elliptical_followup_discovers_new_deep_rank(monkeypatch: Any) -> None:
    """Elliptical follow-up promotes a newly discovered deep-ranked source."""
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Начало про утренний разбор.")
    deep = _record(
        "chapter-9:ru:deep",
        section="chapter-9",
        start=0,
        text="Глубокий ответ ДЕЛЬТА-МАРКЕР рядом.",
    )
    index = _fake_index([first, deep])
    _install_branch(
        monkeypatch,
        [
            {first.chunk_id: _fused(first.chunk_id, 50.0)},
            {
                first.chunk_id: _fused(first.chunk_id, 50.0),
                deep.chunk_id: _fused(deep.chunk_id, 60.0),
            },
        ],
        [[first.chunk_id], [deep.chunk_id]],
    )
    model = _ScriptedModel(
        selections=[
            {
                "selected_chunk_ids": [first.chunk_id],
                "need_more_detail": True,
                "followup_queries": ["уточнение про глубокий ответ"],
            },
            {
                "selected_chunk_ids": [deep.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            },
        ],
        markers={"need-1": "ДЕЛЬТА-МАРКЕР"},
    )
    pack = await aretrieve_with_semantic_selection(
        index,
        ["а подробнее про это"],
        resolved_intent="а подробнее про глубокий ответ",
        conversation_context="утренний разбор",
        user_message="а подробнее?",
        selection_model=model,
        information_needs=[{"need_id": "need-1", "text": "глубокий ответ"}],
        max_iterations=3,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "ready"
    assert int(metadata.get("loop_searches", 0)) >= 1
    served = " ".join(str(p.exact_text) for p in pack.passages)
    assert "ДЕЛЬТА-МАРКЕР" in served


async def test_slow_selector_is_deadline_bounded_with_typed_failure(monkeypatch: Any) -> None:
    """Slow provider yields one bounded typed failure, no retry grind."""
    import time as _time

    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Начало про утренний разбор.")
    index = _fake_index([first])
    _install_branch(
        monkeypatch,
        [{first.chunk_id: _fused(first.chunk_id, 50.0)}],
        [[first.chunk_id]],
    )
    monkeypatch.setattr(retrieval_node_mod, "INTERACTIVE_LATENCY_BUDGET_MS", 120.0)
    monkeypatch.setattr(retrieval_node_mod, "SELECTION_PER_CALL_BUDGET_S", 0.05)
    monkeypatch.setattr(retrieval_node_mod, "COVERAGE_PER_CALL_BUDGET_S", 0.05)
    model = _ScriptedModel(selections=[], markers={"need-1": "НЕДОСТИЖИМЫЙ-МАРКЕР"}, delay_s=5.0)
    started = _time.perf_counter()
    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор тяги"],
        resolved_intent="утренний разбор тяги",
        conversation_context="",
        user_message="как разбирать тягу утром",
        selection_model=model,
        information_needs=[{"need_id": "need-1", "text": "утренний разбор"}],
        max_iterations=3,
    )
    elapsed = _time.perf_counter() - started
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "exhausted"
    assert metadata.get("coverage_exhaustion_reason") == "latency-budget"
    assert metadata.get("coverage_all_covered") is False
    # Bounded: wall time far below the 5s provider tail, single attempt.
    assert elapsed < 2.0
    assert model.calls <= 1
    assert int(metadata.get("loop_model_calls", 0)) <= 1


async def test_rate_limit_propagates_without_swallow(monkeypatch: Any) -> None:
    """Actual 429 escapes the loop for runner retire/resume."""
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection
    from aa.opencode.errors import OpenCodeRateLimitError

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Начало про утренний разбор.")
    index = _fake_index([first])
    _install_branch(
        monkeypatch,
        [{first.chunk_id: _fused(first.chunk_id, 50.0)}],
        [[first.chunk_id]],
    )

    class _RateLimited:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise OpenCodeRateLimitError("429 slow down")

    with pytest.raises(OpenCodeRateLimitError):
        await aretrieve_with_semantic_selection(
            index,
            ["утренний разбор тяги"],
            resolved_intent="утренний разбор тяги",
            conversation_context="",
            user_message="как разбирать тягу утром",
            selection_model=_RateLimited(),
            information_needs=[{"need_id": "need-1", "text": "утренний разбор"}],
            max_iterations=3,
        )


async def test_same_unsuccessful_fingerprint_cannot_loop(monkeypatch: Any) -> None:
    """Identical preview set stops without a second selector call."""
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Начало про утренний разбор.")
    index = _fake_index([first])
    _install_branch(
        monkeypatch,
        [
            {first.chunk_id: _fused(first.chunk_id, 50.0)},
            {first.chunk_id: _fused(first.chunk_id, 50.0)},
        ],
        [[first.chunk_id]],
    )
    model = _ScriptedModel(
        selections=[
            {
                "selected_chunk_ids": [first.chunk_id],
                "need_more_detail": True,
                "followup_queries": ["новый запрос про поддержку"],
            },
            {
                "selected_chunk_ids": [first.chunk_id],
                "need_more_detail": True,
                "followup_queries": ["еще новый запрос про поддержку"],
            },
        ],
        markers={"need-1": "НЕДОСТИЖИМЫЙ-МАРКЕР"},
    )
    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор тяги"],
        resolved_intent="утренний разбор тяги",
        conversation_context="",
        user_message="как разбирать тягу утром",
        selection_model=model,
        information_needs=[{"need_id": "need-1", "text": "утренний разбор"}],
        query_need_map=[{"query_id": "q1", "need_ids": ["need-1"]}],
        max_iterations=3,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "exhausted"
    assert metadata.get("coverage_all_covered") is False
    # Second iteration previews are identical (same pool, no promotion),
    # so no second selector call is issued: one selection + one coverage.
    assert model.calls == 2
    assert int(metadata.get("loop_model_calls", 0)) == 2
    assert "preview-unchanged-skip" in list(metadata.get("loop_progress_events", []))


async def test_adjacent_expand_progress_continues(monkeypatch: Any) -> None:
    """Adjacent canonical expansion that cites a need counts as progress."""
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Начало без ответа.")
    index = _fake_index([first])
    _install_branch(
        monkeypatch,
        [{first.chunk_id: _fused(first.chunk_id, 50.0)}],
        [[first.chunk_id]],
    )
    model = _ScriptedModel(
        selections=[
            {
                "selected_chunk_ids": [first.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            }
        ],
        markers={"need-1": "РАСШИРЕНИЕ-МАРКЕР"},
    )

    import aa.conversation.coverage_loop as coverage_loop_mod

    real_expand = coverage_loop_mod.expand_read_ranges

    def _fake_expand(
        index_arg: Any, read_ids: list[str], *, neighbor_window: int, extra_step: int = 1
    ) -> list[Any]:
        grown = real_expand(
            index_arg, read_ids, neighbor_window=neighbor_window, extra_step=extra_step
        )
        # Adjacent canonical context carries the deciding marker: append
        # it alongside whatever the canonical expansion returned so the
        # grown pack (not a new search hit) closes coverage.
        from aa.retrieval.evidence import EvidencePassageData

        pid = "chapter-1#synthetic:0-10:" + _sha("adjacent")[:16]
        synthetic = EvidencePassageData(
            passage_id=pid,
            exact_text="Соседний контекст РАСШИРЕНИЕ-МАРКЕР рядом.",
            source_id="ru-fourth-edition-txt",
            section_id="chapter-1",
            child_chunk_ids=(first.chunk_id,),
            char_start=0,
            char_end=10,
            text_sha256=_sha("Соседний контекст РАСШИРЕНИЕ-МАРКЕР рядом."),
            source_sha256="s" * 64,
        )
        return [*list(grown or []), synthetic]

    monkeypatch.setattr(coverage_loop_mod, "expand_read_ranges", _fake_expand)
    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор тяги"],
        resolved_intent="утренний разбор тяги",
        conversation_context="",
        user_message="как разбирать тягу утром",
        selection_model=model,
        information_needs=[{"need_id": "need-1", "text": "утренний разбор"}],
        max_iterations=3,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "ready"
    served = " ".join(str(p.exact_text) for p in pack.passages)
    assert "РАСШИРЕНИЕ-МАРКЕР" in served
