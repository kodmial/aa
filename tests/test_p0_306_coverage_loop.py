"""AUDIT P0-4 (kodmial/aa#306): semantic retrieval as a read/coverage loop.

Regressions on invented fixture text (no canonical book text):

- preview looks relevant but full text does not answer; the loop
  searches again (``need_more_detail=false`` never proves sufficiency);
- a needed exception lives in adjacent context; ``expand`` closes
  coverage without requiring a new chunk id;
- a two-part request with only one need covered is detected per need;
- a follow-up search surfaces a candidate previously discovered but not
  read (semantic reconsideration, never rank-only append);
- no-new-ID plus useful expansion counts as progress;
- many-new-IDs plus no improved coverage does not count as progress;
- normal repair and safety repair traverse the same evidence loop;
- exhaustion never issues a positive support certificate.

Plus: lexical signals never mark coverage sufficient, request/control
isolation, narrowing that drops a condition invalidates coverage, and
loop token/call/latency metrics.
"""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace
from typing import Any

from aa.conversation.conversation_context import InformationNeed
from aa.retrieval.fusion import FusedCandidate


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    chunk_id: str,
    *,
    section: str = "chapter-1",
    start: int = 0,
    text: str = "Текст.",
    parent: str | None = None,
    prev: str | None = None,
    next_id: str | None = None,
) -> Any:
    from aa.retrieval.index import ChunkRecord

    end = start + len(text)
    return ChunkRecord(
        chunk_id=chunk_id,
        logical_chunk_id=chunk_id,
        section=section,
        book="aa-big-book",
        parent=parent or ("p-" + chunk_id),
        prev=prev,
        next=next_id,
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


def _fused(chunk_id: str, score: float) -> FusedCandidate:
    return FusedCandidate(
        chunk_id=chunk_id,
        fused_score=score,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=score,
        dense_score=score,
    )


def _needs(*pairs: tuple[str, str]) -> list[InformationNeed]:
    return [InformationNeed(need_id=nid, text=text) for nid, text in pairs]


class _ScriptedLoopModel:
    """One scripted model serving selection and coverage prompts.

    Selection prompts (``<candidates>``) answer from the scripted
    per-call plan; coverage prompts (``<passages>``) judge every need by
    marker presence in the full exact passage texts, citing passages
    that actually contain the marker with the marker itself as the
    verified supporting span.
    """

    def __init__(
        self,
        selections: list[dict[str, Any]],
        markers: dict[str, str],
    ) -> None:
        self._selections = list(selections)
        self._markers = dict(markers)
        self.selection_prompts: list[str] = []
        self.coverage_prompts: list[str] = []
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        self.calls += 1
        if "<passages>" in prompt and "<original_request>" in prompt:
            self.coverage_prompts.append(prompt)
            passage_texts: dict[str, str] = {}
            for match in re.finditer(
                r'<passage id="([^"]+)"[^>]*>(.*?)</passage>', prompt, re.DOTALL
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
        self.selection_prompts.append(prompt)
        if self._selections:
            return dict(self._selections.pop(0))
        return {
            "selected_chunk_ids": [],
            "need_more_detail": False,
            "followup_queries": [],
        }


def _patch_pools(
    monkeypatch: Any, pools: list[dict[str, FusedCandidate]], per_ids: list[list[str]]
) -> dict[str, Any]:
    """Serve canned fused pools per search call (discover, then follow-ups)."""
    from aa.retrieval import evidence as evidence_mod

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


def _pack_texts(pack: Any) -> str:
    return " ".join(p.exact_text for p in pack.passages)


async def test_preview_relevant_but_full_text_insufficient_searches_again(
    monkeypatch: Any,
) -> None:
    """need_more_detail=false never proves sufficiency: full-read miss searches again."""
    misleading = _record(
        "chapter-1:ru:mis",
        start=0,
        text="Многообещающее начало про утренний разбор. Продолжение без ответа.",
    )
    answer = _record(
        "chapter-2:ru:ans",
        section="chapter-2",
        start=0,
        text="ОТВЕТ-А: утром разбирать тягу помогает поддержка рядом.",
    )
    index = _fake_index([misleading, answer])
    first = {misleading.chunk_id: _fused(misleading.chunk_id, 50.0)}
    second = {
        answer.chunk_id: _fused(answer.chunk_id, 60.0),
        misleading.chunk_id: _fused(misleading.chunk_id, 5.0),
    }
    _patch_pools(
        monkeypatch,
        [first, second],
        [[misleading.chunk_id], [answer.chunk_id]],
    )
    model = _ScriptedLoopModel(
        selections=[
            {
                # Preview looked relevant, yet the full text cannot answer.
                "selected_chunk_ids": [misleading.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            },
            {
                "selected_chunk_ids": [answer.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            },
        ],
        markers={"need-1": "ОТВЕТ-А"},
    )
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор тяги"],
        resolved_intent="утренний разбор тяги",
        conversation_context="",
        user_message="как разбирать тягу утром",
        selection_model=model,
        information_needs=_needs(("need-1", "утренний разбор тяги")),
    )
    assert "ОТВЕТ-А" in _pack_texts(pack)
    assert pack.retrieval_metadata["coverage_status"] == "ready"
    assert pack.retrieval_metadata["coverage_all_covered"] is True
    assert pack.retrieval_metadata["loop_searches"] >= 1
    assert pack.retrieval_metadata["loop_model_calls"] >= 2
    assert pack.retrieval_metadata["loop_token_estimate"] > 0
    assert pack.retrieval_metadata["latency_ms"] >= 0.0


async def test_adjacent_exception_expand_closes_without_new_id(monkeypatch: Any) -> None:
    """A needed qualifier adjacent to a read range closes coverage via expand."""
    from aa.retrieval.evidence import RetrievalConfig

    core_text = "ОТВЕТ-А: утром разбирать тягу помогает поддержка рядом. "
    qual_text = "УСЛОВИЕ: если рядом поддержка."
    # No prev/next links: window 0 reads the core chunk alone, so the
    # adjacent qualifier is genuinely unreached until expand runs.
    core = _record("chapter-1:ru:core", start=0, text=core_text)
    qual = _record("chapter-1:ru:qual", start=len(core_text), text=qual_text)
    index = _fake_index([core, qual])
    pool = {core.chunk_id: _fused(core.chunk_id, 10.0)}
    _patch_pools(monkeypatch, [pool], [[core.chunk_id]])

    class _ConditionModel(_ScriptedLoopModel):
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            self.calls += 1
            if "<passages>" in prompt and "<original_request>" in prompt:
                self.coverage_prompts.append(prompt)
                joined = " ".join(
                    " ".join(body.split())
                    for _, body in re.findall(
                        r'<passage id="([^"]+)"[^>]*>(.*?)</passage>', prompt, re.DOTALL
                    )
                )
                if "ОТВЕТ-А" in joined and "УСЛОВИЕ" in joined:
                    ids = re.findall(r'<passage id="([^"]+)"', prompt)
                    return {
                        "needs": [
                            {
                                "need_id": "need-1",
                                "covered": True,
                                "supporting_passage_ids": ids,
                                "supporting_quote": "УСЛОВИЕ",
                                "missing": "",
                            }
                        ]
                    }
                return {
                    "needs": [
                        {
                            "need_id": "need-1",
                            "covered": False,
                            "supporting_passage_ids": [],
                            "supporting_quote": "",
                            "missing": "условие из соседнего отрывка отсутствует",
                        }
                    ]
                }
            self.selection_prompts.append(prompt)
            if self._selections:
                return dict(self._selections.pop(0))
            return {"selected_chunk_ids": [], "need_more_detail": False, "followup_queries": []}

    model = _ConditionModel(
        selections=[
            {
                "selected_chunk_ids": [core.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            }
        ],
        markers={},
    )
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор с условием"],
        config=RetrievalConfig(neighbor_window=0),
        resolved_intent="утренний разбор с условием",
        conversation_context="",
        user_message="как разбирать тягу утром",
        selection_model=model,
        information_needs=_needs(("need-1", "утренний разбор с условием")),
    )
    assert pack.retrieval_metadata["coverage_status"] == "ready"
    assert pack.retrieval_metadata["loop_expansions"] >= 1
    assert pack.retrieval_metadata["loop_searches"] == 0
    assert pack.retrieval_metadata["read_ids"] == [core.chunk_id]
    assert "УСЛОВИЕ" in _pack_texts(pack)


async def test_two_part_request_reports_per_need_coverage(monkeypatch: Any) -> None:
    """Two needs: only the evidenced one is covered; the other stays explicit."""
    answer_a = _record(
        "chapter-1:ru:a",
        start=0,
        text="ОТВЕТ-А: утром разбирать тягу помогает поддержка рядом.",
    )
    answer_b = _record(
        "chapter-7:ru:b",
        section="chapter-7",
        start=0,
        text="ОТВЕТ-Б: вечером сообщество поддерживает спокойно.",
    )
    index = _fake_index([answer_a, answer_b])
    first = {answer_a.chunk_id: _fused(answer_a.chunk_id, 20.0)}
    second = {
        answer_a.chunk_id: _fused(answer_a.chunk_id, 20.0),
        answer_b.chunk_id: _fused(answer_b.chunk_id, 25.0),
    }
    _patch_pools(
        monkeypatch,
        [first, second],
        [[answer_a.chunk_id], [answer_b.chunk_id]],
    )
    model = _ScriptedLoopModel(
        selections=[
            {
                "selected_chunk_ids": [answer_a.chunk_id],
                "need_more_detail": True,
                "followup_queries": ["вечерняя поддержка сообщества"],
            },
            {
                "selected_chunk_ids": [answer_a.chunk_id, answer_b.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            },
        ],
        markers={"need-1": "ОТВЕТ-А", "need-2": "ОТВЕТ-Б"},
    )
    # Observe the partial state directly: one need covered, one missing.
    from aa.conversation.coverage_loop import aassess_coverage
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection
    from aa.retrieval.evidence import expand_small_to_big

    first_expanded = expand_small_to_big(index, [_fused(answer_a.chunk_id, 1.0)])
    partial = await aassess_coverage(
        list(first_expanded),
        _needs(("need-1", "утро"), ("need-2", "вечер")),
        original_request="утро и вечер",
        model=model,
    )
    by_id = {item.need_id: item for item in partial.needs}
    assert by_id["need-1"].covered is True
    assert by_id["need-2"].covered is False
    assert partial.all_covered is False
    first_spans = list(by_id["need-1"].supporting_spans or ())
    assert any("ОТВЕТ-А" in span for span in first_spans)
    assert by_id["need-2"].missing != ""

    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор", "вечерняя поддержка"],
        resolved_intent="утренний разбор и вечерняя поддержка",
        conversation_context="",
        user_message="как утром и вечером",
        selection_model=model,
        information_needs=_needs(("need-1", "утро"), ("need-2", "вечер")),
    )
    assert pack.retrieval_metadata["coverage_status"] == "ready"
    per_need = {item["need_id"]: item for item in pack.retrieval_metadata["coverage_per_need"]}
    assert per_need["need-1"]["covered"] is True
    assert per_need["need-2"]["covered"] is True
    assert per_need["need-1"]["anchors"]
    assert "ОТВЕТ-Б" in _pack_texts(pack)


async def test_followup_surfaces_discovered_but_unread_candidate(monkeypatch: Any) -> None:
    """A rank-66 candidate discovered but never read is reconsidered after search."""
    records: list[Any] = []
    for pos in range(30):
        records.append(
            _record(f"chapter-1:ru:f{pos:04d}", start=pos * 100, text=f"Фоновый отрывок {pos}.")
        )
    target = _record(
        "chapter-7:ru:deep",
        section="chapter-7",
        start=9000,
        text="ОТВЕТ-ГЛУБОКИЙ: решающий отрывок про разбор.",
    )
    records.append(target)
    index = _fake_index(records)
    first = {
        record.chunk_id: _fused(record.chunk_id, float(100 - pos))
        for pos, record in enumerate(records)
    }
    second = {
        target.chunk_id: _fused(target.chunk_id, 200.0),
        records[0].chunk_id: _fused(records[0].chunk_id, 1.0),
    }
    per_a = [record.chunk_id for record in records]
    _patch_pools(monkeypatch, [first, second], [per_a, [target.chunk_id]])
    model = _ScriptedLoopModel(
        selections=[
            {
                "selected_chunk_ids": [records[0].chunk_id],
                "need_more_detail": True,
                "followup_queries": ["решающий разбор"],
            },
            {
                "selected_chunk_ids": [target.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            },
        ],
        markers={"need-1": "ОТВЕТ-ГЛУБОКИЙ"},
    )
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    pack = await aretrieve_with_semantic_selection(
        index,
        ["разбор"],
        resolved_intent="решающий разбор",
        conversation_context="",
        user_message="решающий разбор",
        selection_model=model,
        information_needs=_needs(("need-1", "решающий разбор")),
    )
    assert target.chunk_id in pack.retrieval_metadata["discovered_ids"]
    assert target.chunk_id in pack.retrieval_metadata["read_ids"]
    assert pack.retrieval_metadata["followup_added"] >= 1
    assert "ОТВЕТ-ГЛУБОКИЙ" in _pack_texts(pack)


async def test_no_new_id_useful_expansion_counts_as_progress(monkeypatch: Any) -> None:
    """Same IDs plus a need-closing expansion is progress, not stagnation."""
    core_text = "ОТВЕТ-А: утром разбирать тягу помогает поддержка рядом. "
    qual_text = "УСЛОВИЕ-РЯДОМ: исключение рядом в соседнем отрывке."
    # No prev/next links: window 0 reads the core chunk alone.
    core = _record("chapter-1:ru:core", start=0, text=core_text)
    qual = _record("chapter-1:ru:qual", start=len(core_text), text=qual_text)
    index = _fake_index([core, qual])
    pool = {core.chunk_id: _fused(core.chunk_id, 10.0)}
    _patch_pools(monkeypatch, [pool], [[core.chunk_id]])

    class _ConditionCoverageModel(_ScriptedLoopModel):
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            self.calls += 1
            if "<passages>" in prompt and "<original_request>" in prompt:
                self.coverage_prompts.append(prompt)
                bodies = [
                    " ".join(body.split())
                    for _, body in re.findall(
                        r'<passage id="([^"]+)"[^>]*>(.*?)</passage>', prompt, re.DOTALL
                    )
                ]
                joined = " ".join(bodies)
                if "ОТВЕТ-А" in joined and "УСЛОВИЕ-РЯДОМ" in joined:
                    ids = re.findall(r'<passage id="([^"]+)"', prompt)
                    return {
                        "needs": [
                            {
                                "need_id": "need-1",
                                "covered": True,
                                "supporting_passage_ids": ids,
                                "supporting_quote": "УСЛОВИЕ-РЯДОМ",
                                "missing": "",
                            }
                        ]
                    }
                return {
                    "needs": [
                        {
                            "need_id": "need-1",
                            "covered": False,
                            "supporting_passage_ids": [],
                            "supporting_quote": "",
                            "missing": "условие из соседнего отрывка отсутствует",
                        }
                    ]
                }
            self.selection_prompts.append(prompt)
            if self._selections:
                return dict(self._selections.pop(0))
            return {"selected_chunk_ids": [], "need_more_detail": False, "followup_queries": []}

    from aa.retrieval.evidence import RetrievalConfig

    model = _ConditionCoverageModel(
        selections=[
            {
                "selected_chunk_ids": [core.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            }
        ],
        markers={},
    )
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    pack = await aretrieve_with_semantic_selection(
        index,
        ["разбор с условием"],
        config=RetrievalConfig(neighbor_window=0),
        resolved_intent="разбор с условием",
        conversation_context="",
        user_message="разбор с условием",
        selection_model=model,
        information_needs=_needs(("need-1", "разбор с условием")),
    )
    assert pack.retrieval_metadata["coverage_status"] == "ready"
    assert pack.retrieval_metadata["read_ids"] == [core.chunk_id]
    assert pack.retrieval_metadata["loop_searches"] == 0
    assert any(
        event.startswith("expand-") for event in pack.retrieval_metadata["loop_progress_events"]
    )
    assert "УСЛОВИЕ-РЯДОМ" in _pack_texts(pack)


async def test_many_new_ids_without_coverage_gain_is_not_progress(monkeypatch: Any) -> None:
    """Novel irrelevant ids alone never count as semantic progress."""
    junk = [
        _record(
            f"chapter-9:ru:j{pos:04d}", section="chapter-9", start=pos * 100, text="Погодный фон."
        )
        for pos in range(25)
    ]
    index = _fake_index(junk)
    first = {
        record.chunk_id: _fused(record.chunk_id, float(50 - pos)) for pos, record in enumerate(junk)
    }
    second = {
        record.chunk_id: _fused(record.chunk_id, float(60 - pos)) for pos, record in enumerate(junk)
    }
    _patch_pools(
        monkeypatch,
        [first, second, second],
        [[record.chunk_id for record in junk]] * 3,
    )
    model = _ScriptedLoopModel(
        selections=[
            {
                "selected_chunk_ids": [junk[0].chunk_id],
                "need_more_detail": True,
                "followup_queries": ["другой погодный запрос"],
            },
            {
                "selected_chunk_ids": [junk[1].chunk_id],
                "need_more_detail": True,
                "followup_queries": ["еще погодный запрос"],
            },
            {
                "selected_chunk_ids": [junk[2].chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            },
        ],
        markers={"need-1": "НЕСУЩЕСТВУЮЩИЙ-МАРКЕР"},
    )
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    pack = await aretrieve_with_semantic_selection(
        index,
        ["разбор тяги"],
        resolved_intent="разбор тяги",
        conversation_context="",
        user_message="разбор тяги",
        selection_model=model,
        information_needs=_needs(("need-1", "разбор тяги")),
        max_iterations=3,
    )
    assert pack.retrieval_metadata["fused_unique"] >= 20
    assert pack.retrieval_metadata["coverage_status"] == "exhausted"
    assert pack.retrieval_metadata["coverage_exhausted"] is True
    assert pack.retrieval_metadata["coverage_all_covered"] is False
    assert pack.retrieval_metadata["coverage_missing_need_ids"] == ["need-1"]
    assert not any(
        event.startswith("need-covered")
        for event in pack.retrieval_metadata["loop_progress_events"]
    )


async def test_normal_and_safety_repair_share_one_evidence_loop(monkeypatch: Any) -> None:
    """Ordinary and safety recovery retrieval call the same shared mechanism.

    Both repair kinds resolve to the single shared async
    discover/select/read/assess/expand/search_more function (never a
    legacy synchronous rank-only bypass), with the original request
    isolated from repair/safety control text.
    """
    from langchain_core.messages import AIMessage

    from aa.conversation import retrieval_node as retrieval_node_mod
    from aa.conversation.response_units import split_response_units
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.safety.outbound import SAFE_RECOVERY_INSTRUCTION
    from tests.test_p0_4_natural_grounding import (
        _AnswerModel as _ProvenAnswer,
    )
    from tests.test_p0_4_natural_grounding import (
        _pack_entry as _proven_pack_entry,
    )
    from tests.test_p0_4_natural_grounding import (
        _twelve_queries as _proven_twelve_queries,
    )
    from tests.test_p0_4_natural_grounding import (
        _VerifierModel as _ProvenVerifier,
    )
    from tests.test_p0_300_safe_recovery import _HARMFUL_DRINK_TEST as _HARMFUL

    shared_fn = retrieval_node_mod.aretrieve_with_semantic_selection
    assert "coverage" in (shared_fn.__doc__ or "").casefold()
    calls: list[dict[str, Any]] = []

    async def _spy_shared(index: Any, queries: Any, **kwargs: Any) -> Any:
        from aa.retrieval.evidence import EvidencePack, EvidencePassageData

        calls.append(
            {
                "func": retrieval_node_mod.aretrieve_with_semantic_selection,
                "queries": list(queries),
                "kwargs": dict(kwargs),
            }
        )
        text = str(_spy_shared_text)
        passage = EvidencePassageData(
            passage_id="chapter-3#exp0001",
            exact_text=text,
            source_id="ru-fourth-edition-txt",
            section_id="chapter-3",
            child_chunk_ids=("chapter-3:ru:1",),
            char_start=120,
            char_end=120 + len(text),
            text_sha256=_sha(text),
            source_sha256="s" * 64,
        )
        return EvidencePack(
            passages=(passage,), total_tokens=10, corpus_version="v", retrieval_metadata={}
        )

    _spy_shared_text = "Фиктивная поддержка рядом помогает пережить тягу спокойно."
    monkeypatch.setattr(retrieval_node_mod, "aretrieve_with_semantic_selection", _spy_shared)

    # --- Ordinary unsupported-answer repair (proven #145/#208 fixture).
    first_pack = [_proven_pack_entry(passage_id="chapter-3#exp0000", char_start=0, char_end=120)]
    first_draft = "Поддержка рядом помогает. Тяга лечится луной за вечер."
    repaired_draft = "Поддержка рядом помогает пережить тягу спокойно."
    repaired_units = split_response_units(repaired_draft)
    planner_inputs: list[str] = []

    async def _fake_planner(
        text: str, *, model: Any = None, summary: str = "", recent: Any = None, **kwargs: Any
    ) -> Any:
        _ = (model, summary, recent, kwargs)
        planner_inputs.append(text)

        class _Plan:
            queries = _proven_twelve_queries()

        return _Plan()

    monkeypatch.setattr("aa.conversation.planner_node.run_planner", _fake_planner)
    live_request = "что помогает при тяге?"
    outcome = await run_v2_answer_turn(
        user_message=live_request,
        summary="",
        recent=[],
        evidence_pack=first_pack,
        answer_model=_ProvenAnswer([first_draft, repaired_draft]),
        verifier_model=_ProvenVerifier(
            [
                {
                    "units": [
                        {
                            "unit_id": "u1",
                            "scope": "book",
                            "supported": True,
                            "evidence_passage_ids": [first_pack[0]["passage_id"]],
                        },
                        {
                            "unit_id": "u2",
                            "scope": "book",
                            "supported": False,
                            "evidence_passage_ids": [first_pack[0]["passage_id"]],
                        },
                    ],
                    "all_required_supported": False,
                },
                {
                    "units": [
                        {
                            "unit_id": unit.unit_id,
                            "scope": "book",
                            "supported": True,
                            "evidence_passage_ids": ["chapter-3#exp0001"],
                        }
                        for unit in repaired_units
                    ],
                    "all_required_supported": True,
                },
            ]
        ),
        planner_model=object(),
        retrieval_index=object(),
    )
    assert outcome["text"] == repaired_draft
    assert outcome["rounds"] == 1
    # The planner re-queries the original request, never the unsupported
    # draft claim (request isolation carried from #300 to ordinary repair).
    assert planner_inputs, "ordinary repair must re-plan"
    for seen in planner_inputs:
        assert seen.endswith(live_request)
        assert "луной" not in seen
    ordinary_calls = [call for call in calls if call["kwargs"].get("repair_hint")]
    assert ordinary_calls, "ordinary repair must traverse the shared loop with a hint"
    for call in ordinary_calls:
        assert call["func"] is retrieval_node_mod.aretrieve_with_semantic_selection
        assert "луной" in str(call["kwargs"].get("repair_hint", ""))
        assert "луной" not in " ".join(call["queries"])

    # --- Safety recovery phase 2 reaches the same shared function.
    calls.clear()
    safe_pack = [_proven_pack_entry()]

    class _AlwaysHarmfulAnswer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content=_HARMFUL)

    class _SupportiveVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [safe_pack[0]["passage_id"]],
                "addresses_intent": True,
            }

    import pytest as _pytest

    from aa.conversation.failures import TurnFailed as _TurnFailed

    with _pytest.raises(_TurnFailed) as _exc_info:
        await run_v2_answer_turn(
            user_message="Вечером тяжело без выпивки, как справляться?",
            summary="",
            recent=[],
            evidence_pack=safe_pack,
            answer_model=_AlwaysHarmfulAnswer(),
            verifier_model=_SupportiveVerifier(),
            planner_model=object(),
            retrieval_index=object(),
            initial_query_count=12,
            planner_mode="retrieval",
            resolved_intent="Вечером тяжело без выпивки, как справляться?",
        )
    assert _exc_info.value.category == "safety-blocked"
    assert calls, "safety recovery must traverse the same shared loop"
    for call in calls:
        assert call["func"] is retrieval_node_mod.aretrieve_with_semantic_selection
        joined = " ".join(call["queries"])
        assert SAFE_RECOVERY_INSTRUCTION not in joined
        assert "инструкция" not in joined.casefold()
        assert _HARMFUL[:20] not in joined


async def test_exhaustion_never_issues_positive_certificate(monkeypatch: Any) -> None:
    """An exhausted loop with insufficient evidence cannot certify an answer."""
    junk = _record("chapter-9:ru:only", section="chapter-9", start=0, text="Погодный фон.")
    index = _fake_index([junk])
    pool = {junk.chunk_id: _fused(junk.chunk_id, 9.0)}
    _patch_pools(monkeypatch, [pool, pool], [[junk.chunk_id], [junk.chunk_id]])
    model = _ScriptedLoopModel(
        selections=[
            {
                "selected_chunk_ids": [junk.chunk_id],
                "need_more_detail": True,
                "followup_queries": ["другой запрос"],
            },
            {
                "selected_chunk_ids": [junk.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            },
        ],
        markers={"need-1": "НЕСУЩЕСТВУЮЩИЙ-МАРКЕР"},
    )
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    pack = await aretrieve_with_semantic_selection(
        index,
        ["разбор тяги"],
        resolved_intent="разбор тяги",
        conversation_context="",
        user_message="разбор тяги",
        selection_model=model,
        information_needs=_needs(("need-1", "разбор тяги")),
        max_iterations=2,
    )
    assert pack.retrieval_metadata["coverage_status"] == "exhausted"
    assert pack.retrieval_metadata["coverage_exhaustion_reason"] in (
        "repeated-state",
        "iteration-budget",
        "no-followup-queries",
        "search-budget",
    )
    assert pack.retrieval_metadata["loop_iterations"] >= 1
    # Even with a substantive draft, finalization fails closed: no claim
    # verdicts exist for insufficient evidence, so no positive support
    # certificate is ever issued.
    from aa.conversation.finalization import FinalizationError, certify_candidate
    from aa.conversation.retrieval_node import pack_to_state

    _, pack_dicts = pack_to_state(pack)
    candidate_text = "Разбор тяги помогает утром."
    from aa.conversation.finalization import AnswerCandidate

    candidate = AnswerCandidate(
        text=candidate_text,
        evidence_bundle=pack_dicts,
        context_digest="d" * 64,
        outcome_kind="answer",
    )
    try:
        certify_candidate(
            candidate=candidate,
            grounding_result=None,
            question="как разбирать тягу",
            resolved_intent="разбор тяги",
            summary="",
            recent_messages=[],
        )
    except FinalizationError as exc:
        assert exc.category in ("missing-verdicts", "context-mismatch", "claim-missing")
    else:
        raise AssertionError("exhausted evidence must never certify")


def _merge_pack_dict(passage_id: str, text: str, **extra: Any) -> dict[str, Any]:
    return {
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
        **extra,
    }


def test_low_overlap_relevant_displaces_lexically_higher_pack() -> None:
    """#306 carry-forward: lexical scoring never decides budget survival.

    A full pack of high-overlap decoys cannot keep out a newly relevant
    low-overlap passage for a represented need; per-need round-robin
    (not lexical order) decides, and a lexically higher decoy is the one
    displaced.
    """
    from aa.conversation.turn_pipeline import MAX_PACK_PASSAGES, merge_pack_dicts

    intent = "трезвый утренний разбор тяги поддержка"
    old = [
        _merge_pack_dict(
            f"chapter-1#decoy-{index}", f"трезвый утренний разбор тяги поддержка {index}"
        )
        for index in range(MAX_PACK_PASSAGES)
    ]
    fresh_text = "Вечером рядом сообщество."
    fresh = [_merge_pack_dict("chapter-7#fresh-relevant", fresh_text, need_ids=["need-2"])]
    for entry in old:
        entry["need_ids"] = ["need-1"]
    merged = merge_pack_dicts(
        old,
        fresh,
        resolved_intent=intent,
        conversation_context="",
        information_needs=[
            {"need_id": "need-1", "text": intent},
            {"need_id": "need-2", "text": "вечерняя поддержка сообщества"},
        ],
    )
    ids = {item["passage_id"] for item in merged}
    assert "chapter-7#fresh-relevant" in ids
    assert len(merged) == MAX_PACK_PASSAGES
    assert len(ids) == MAX_PACK_PASSAGES


def test_lexical_signals_never_mark_coverage_sufficient() -> None:
    """Without a model, full lexical overlap still reports uncovered."""
    import asyncio as _asyncio

    from aa.conversation.coverage_loop import aassess_coverage
    from aa.retrieval.evidence import EvidencePassageData

    text = "утренний разбор тяги поддержка сообщество рядом"
    passage = EvidencePassageData(
        passage_id="chapter-1#lex",
        exact_text=text,
        source_id="ru-fourth-edition-txt",
        section_id="chapter-1",
        child_chunk_ids=("chapter-1:ru:lex",),
        char_start=0,
        char_end=len(text),
        text_sha256=_sha(text),
        source_sha256="s" * 64,
    )
    verdict = _asyncio.run(
        aassess_coverage(
            [passage],
            _needs(("need-1", "утренний разбор тяги поддержка")),
            original_request="утренний разбор тяги поддержка",
            model=None,
        )
    )
    assert verdict.all_covered is False
    assert verdict.used_model is False
    assert verdict.missing_need_ids == ("need-1",)


def test_followup_sanitizer_drops_repeats_and_control_echoes() -> None:
    """Changed query strings that repeat sent candidates are not progress."""
    from aa.conversation.coverage_loop import loop_fingerprint, sanitize_followup_queries

    seen = {"fp-sent"}
    assert sanitize_followup_queries([], seen_fingerprints=seen) == []
    kept = sanitize_followup_queries(
        ["  разбор тяги  ", "разбор тяги", ""],
        seen_fingerprints=set(),
    )
    assert kept == ["разбор тяги"]
    blocked = sanitize_followup_queries(
        ["разбор тяги", "NSUPP CLAIMXYZ"],
        seen_fingerprints=set(),
        forbidden_texts=["NSUPP CLAIMXYZ"],
    )
    assert blocked == ["разбор тяги"]
    first = loop_fingerprint(
        read_passage_ids=["p1"],
        read_ranges=["0-10"],
        uncovered_need_ids=["need-1"],
        action="select/read/assess",
        candidate_identity="c1",
    )
    second = loop_fingerprint(
        read_passage_ids=["p1"],
        read_ranges=["0-10"],
        uncovered_need_ids=["need-1"],
        action="select/read/assess",
        candidate_identity="c1",
    )
    other = loop_fingerprint(
        read_passage_ids=["p1", "p2"],
        read_ranges=["0-10", "10-20"],
        uncovered_need_ids=[],
        action="select/read/assess",
        candidate_identity="c2",
    )
    assert first == second
    assert first != other


async def test_compiled_graph_retrieval_answer_finalize_identity(
    monkeypatch: Any,
) -> None:
    """Real compiled-graph route: loop retrieval, answer, exact certification.

    The full graph (planner, shared read/coverage retrieval, answer with
    bounded repair, #304 finalizer) serves a certified answer whose
    evidence digest matches the served bundle byte-for-byte.
    """
    import json as _json

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda

    from aa.conversation.finalization import evidence_digest_for_pack
    from aa.conversation.graph import build_turn_graph, turn_input

    question = "что помогает при тяге вечером?"
    intent = "что помогает при тяге вечером"
    chunk_text = "ОТВЕТ-А: поддержка рядом помогает пережить тягу спокойно сегодня."
    ans_chunk = _record("chapter-3:ru:ans", section="chapter-3", start=0, text=chunk_text)
    index = _fake_index([ans_chunk])
    pool = {ans_chunk.chunk_id: _fused(ans_chunk.chunk_id, 30.0)}
    _patch_pools(monkeypatch, [pool], [[ans_chunk.chunk_id]])
    draft = "Поддержка рядом помогает пережить тягу спокойно."

    async def _multi_role(messages: Any) -> Any:
        from langchain_core.messages import BaseMessage as _BM

        text = " ".join(
            str(getattr(item, "content", ""))
            for item in (messages if isinstance(messages, list) else [messages])
            if isinstance(item, _BM)
        )
        if "<candidates>" in text:
            assert "chapter-3:ru:ans" in text
            return AIMessage(
                content=_json.dumps(
                    {
                        "selected_chunk_ids": ["chapter-3:ru:ans"],
                        "need_more_detail": False,
                        "followup_queries": [],
                    }
                )
            )
        if "<passages>" in text and "<original_request>" in text:
            ids = re.findall(r'<passage id="([^"]+)"', text)
            assert ids and "ОТВЕТ-А" in text
            return AIMessage(
                content=_json.dumps(
                    {
                        "needs": [
                            {
                                "need_id": "need-1",
                                "covered": True,
                                "supporting_passage_ids": ids,
                                "supporting_quote": "ОТВЕТ-А",
                                "missing": "",
                            }
                        ]
                    }
                )
            )
        return {"mode": "retrieval", "resolved_intent": intent, "queries": [intent]}

    planner_selection_model = RunnableLambda(_multi_role)

    class _GraphAnswer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content=draft)

    class _GraphVerifier:
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

    graph = build_turn_graph(
        planner_model=planner_selection_model,
        retrieval_index=index,
        answer_model=_GraphAnswer(),
        verifier_model=_GraphVerifier(),
    )
    result = await graph.ainvoke(turn_input(question))
    assert result["final_response"] == draft
    assert result["delivery_status"] == "provisional-certified"
    served_pack = [dict(item) for item in result["evidence_pack"]]
    assert served_pack, "the loop must serve exact evidence"
    assert any("ОТВЕТ-А" in str(item.get("text", "")) for item in served_pack)
    assert all("#" in str(item.get("passage_id", "")) for item in served_pack)
    candidate = result["answer_candidate"]
    certificate = result["verification_certificate"]
    assert certificate["answer_sha256"] == _sha(draft)
    assert certificate["context_digest"] == candidate["context_digest"]
    assert certificate["evidence_digest"] == evidence_digest_for_pack(served_pack)
    assert certificate["whole_answer_verdict"]["supported"] is True
    _ = HumanMessage


def test_narrowing_that_drops_condition_invalidates_coverage() -> None:
    """An expanded passage narrowed back to a child without the condition is incomplete."""
    import asyncio as _asyncio

    from aa.conversation.coverage_loop import aassess_coverage
    from aa.retrieval.evidence import EvidencePassageData

    full_text = "ОТВЕТ-А: утром разбирать тягу помогает поддержка. УСЛОВИЕ: если рядом поддержка."
    child_text = "ОТВЕТ-А: утром разбирать тягу помогает поддержка."

    def _passage(pid: str, text: str) -> EvidencePassageData:
        return EvidencePassageData(
            passage_id=pid,
            exact_text=text,
            source_id="ru-fourth-edition-txt",
            section_id="chapter-1",
            child_chunk_ids=("chapter-1:ru:x",),
            char_start=0,
            char_end=len(text),
            text_sha256=_sha(text),
            source_sha256="s" * 64,
        )

    class _ConditionModel:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            ids = re.findall(r'<passage id="([^"]+)"', prompt)
            joined = " ".join(
                body
                for _, body in re.findall(
                    r'<passage id="([^"]+)"[^>]*>(.*?)</passage>', prompt, re.DOTALL
                )
            )
            covered = "ОТВЕТ-А" in joined and "УСЛОВИЕ" in joined
            return {
                "needs": [
                    {
                        "need_id": "need-1",
                        "covered": covered,
                        "supporting_passage_ids": ids if covered else [],
                        "supporting_quote": "УСЛОВИЕ" if covered else "",
                        "missing": "" if covered else "условие потеряно при сужении",
                    }
                ]
            }

    needs = _needs(("need-1", "разбор с условием"))
    full = _asyncio.run(
        aassess_coverage(
            [_passage("chapter-1#full", full_text)],
            needs,
            original_request="разбор с условием",
            model=_ConditionModel(),
        )
    )
    narrowed = _asyncio.run(
        aassess_coverage(
            [_passage("chapter-1#narrow", child_text)],
            needs,
            original_request="разбор с условием",
            model=_ConditionModel(),
        )
    )
    assert full.all_covered is True
    assert narrowed.all_covered is False
    assert narrowed.missing_need_ids == ("need-1",)
