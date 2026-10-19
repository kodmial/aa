"""P0 SYSTEMIC kodmial/aa#387: retrieval preserves the downstream slice.

Systemic failure class (Gate C ``live-book-grounding-substantive-drinking-2``
on ``live-production-path`` across distinct main SHAs): the sequential
planner -> retrieval -> answer -> verifier pipeline recurred with retrieval
p95 ~45s / max ~65s, answer p50 ~13s, verifier p50 ~17s, repair_turns=0,
answer_rounds=22, budget_exceeded=3 and clarifications=16. The shared turn
budget reserved only 5s for downstream, so retrieval semantic work (up to 8
model rounds) consumed the budget and the answer/verifier stages skipped via
their minimal-slice guards, serving clarification that Gate C counts as
grounding failure. Narrow predicate, safety-clause, narrowing-allowlist,
telemetry, judge-wording and per-stage timeout patches each certified one
surface while the upstream starvation mechanism remained.

Architecture repair (generic, turn-independent, no exact-question branches,
no keyword/stem tables, no step parsing, no token-overlap relevance, no
handcrafted semantic predicates; Product Contract #110, Gate C semantics,
grounding, relevance, safety and Gate E SLO unchanged): retrieval semantic
work stops issuing new model calls once the remaining turn budget cannot
cover one bounded answer attempt plus verifier minimum plus finalization
(40s derived from existing pipeline slices), and caps total semantic rounds.
Exhaustion keeps the best-effort lexical pack with typed
coverage_status=exhausted (never empty when lexical candidates exist, never
fake-covered), so the answer stage still attempts grounded generation.

All strings below are synthetic fixtures for this failure class, never the
literal qualification prompts.
"""

from __future__ import annotations

import hashlib
import pathlib
from types import SimpleNamespace
from typing import Any


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    chunk_id: str,
    *,
    section: str = "chapter-3",
    start: int = 0,
    text: str = "Synthetic passage about a calm evening routine nearby.",
) -> Any:
    from aa.retrieval.index import ChunkRecord

    end = start + len(text)
    return ChunkRecord(
        chunk_id=chunk_id,
        logical_chunk_id=chunk_id,
        section=section,
        book="aa-big-book",
        parent=("p-" + chunk_id),
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


def _install_branch(
    monkeypatch: Any, pools: list[dict[str, Any]], per_ids: list[list[str]]
) -> None:
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


class _CountingModel:
    """Semantic model double counting every structured call."""

    def __init__(self, *, select_ids: list[str]) -> None:
        self._select_ids = list(select_ids)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        self.calls += 1
        if "<passages>" in prompt and "<original_request>" in prompt:
            return {
                "needs": [
                    {
                        "need_id": "need-1",
                        "covered": False,
                        "supporting_passage_ids": [],
                        "supporting_quote": "",
                        "missing": "synthetic shortfall",
                    }
                ]
            }
        return {
            "selected_chunk_ids": list(self._select_ids),
            "need_more_detail": True,
            "followup_queries": ["synthetic follow-up detail"],
        }


class _NeverCalledModel:
    """Fails the test if any semantic call is issued."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        raise AssertionError("no retrieval model call may start without a downstream slice")


def test_downstream_reserve_constants_preserve_slo() -> None:
    from aa.conversation import retrieval_node as node_mod
    from aa.conversation import turn_budget as budget_mod
    from aa.conversation import turn_pipeline as pipeline_mod
    from aa.qualification import self_proving as proving_mod

    assert node_mod.RETRIEVAL_SEMANTIC_DOWNSTREAM_RESERVE_S == 40.0
    assert node_mod.RETRIEVAL_SEMANTIC_MAX_MODEL_CALLS == 4
    # #400: planner+retrieval semantic work fits the Gate E p95 headroom
    # left after the downstream reserve (60s - 40s = 20s).
    assert node_mod.RETRIEVAL_SEMANTIC_STAGE_BUDGET_S == 20.0
    # Authoritative SLO and turn budgets are unchanged by this repair.
    assert budget_mod.TURN_END_TO_END_BUDGET_S == 105.0
    assert pipeline_mod.TURN_END_TO_END_BUDGET_S == 105.0
    assert budget_mod.TURN_HARD_SLO_MS == 120_000.0
    assert budget_mod.TURN_P95_TARGET_MS == 60_000.0
    assert proving_mod.ORDINARY_TURN_BUDGET_MS == 120_000
    assert proving_mod.P95_TARGET_MS == 60_000
    # The new reserve strictly preserves a feasible downstream slice,
    # unlike the 5s per-call reserve that starved answer/verifier.
    assert node_mod.RETRIEVAL_SEMANTIC_DOWNSTREAM_RESERVE_S > budget_mod.DOWNSTREAM_MIN_RESERVE_S


async def test_no_semantic_round_when_downstream_headroom_is_spent(
    monkeypatch: Any,
) -> None:
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection
    from aa.conversation.turn_budget import new_turn_budget

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-3:ru:first", text="Synthetic calm routine ALPHA marker nearby.")
    second = _record("chapter-3:ru:second", text="Synthetic steady support BETA marker nearby.")
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
    # Upstream planner cost already spent 70s: only ~35s remain, below the
    # 40s downstream reserve and far below the 60s Gate E p95 target.
    # #400 closes the old first-pair exemption: no semantic model call
    # may start when it cannot preserve the downstream slice.
    budget = new_turn_budget(upstream_latency_ms=70_000.0)
    model = _CountingModel(select_ids=[first.chunk_id])
    pack = await aretrieve_with_semantic_selection(
        index,
        ["synthetic calm routine query"],
        resolved_intent="synthetic calm routine intent",
        conversation_context="",
        user_message="synthetic calm routine request",
        selection_model=model,
        information_needs=[{"need_id": "need-1", "text": "synthetic calm routine"}],
        max_iterations=3,
        turn_budget=budget,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "exhausted"
    # Closed-vocabulary reason stays latency-budget (fail-closed exhausted);
    # the downstream-reserve cause travels as a progress event.
    assert metadata.get("coverage_exhaustion_reason") == "latency-budget"
    assert metadata.get("coverage_all_covered") is False
    # No semantic call started, so the best-effort lexical pack never
    # starved the answer/verifier reserve.
    assert model.calls == 0
    assert "retrieval-downstream-reserve-stop" in list(
        metadata.get("loop_progress_events", []) or []
    )


async def test_first_slow_semantic_call_is_bounded_by_p95_headroom(
    monkeypatch: Any,
) -> None:
    import asyncio
    import time as _time

    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-3:ru:first", text="Synthetic calm routine ALPHA marker nearby.")
    index = _fake_index([first])
    _install_branch(
        monkeypatch,
        [{first.chunk_id: _fused(first.chunk_id, 50.0)}],
        [[first.chunk_id]],
    )
    # Shrink only the stage headroom so the bounded tail is observable
    # without a multi-second test; the production value is 20s.
    monkeypatch.setattr(retrieval_node_mod, "RETRIEVAL_SEMANTIC_STAGE_BUDGET_S", 0.3)

    class _SlowModel:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            await asyncio.sleep(5.0)
            return {"selected_chunk_ids": [], "need_more_detail": False, "followup_queries": []}

    model = _SlowModel()
    started = _time.perf_counter()
    pack = await aretrieve_with_semantic_selection(
        index,
        ["synthetic calm routine query"],
        resolved_intent="synthetic calm routine intent",
        conversation_context="",
        user_message="synthetic calm routine request",
        selection_model=model,
        information_needs=[{"need_id": "need-1", "text": "synthetic calm routine"}],
        max_iterations=3,
    )
    elapsed = _time.perf_counter() - started
    metadata = dict(pack.retrieval_metadata or {})
    assert elapsed < 2.0
    assert model.calls == 1
    assert metadata.get("coverage_status") == "exhausted"
    assert metadata.get("coverage_exhaustion_reason") == "latency-budget"
    assert metadata.get("typed_stage") == "provider_semantic_timeout"
    assert "selection-deadline-stop" in list(metadata.get("loop_progress_events", []) or [])


async def test_semantic_rounds_capped_despite_ample_budget(monkeypatch: Any) -> None:
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-3:ru:first", text="Synthetic calm routine ALPHA marker nearby.")
    second = _record("chapter-3:ru:second", text="Synthetic steady support BETA marker nearby.")
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
    model = _CountingModel(select_ids=[first.chunk_id])
    pack = await aretrieve_with_semantic_selection(
        index,
        ["synthetic calm routine query"],
        resolved_intent="synthetic calm routine intent",
        conversation_context="",
        user_message="synthetic calm routine request",
        selection_model=model,
        information_needs=[{"need_id": "need-1", "text": "synthetic calm routine"}],
        max_iterations=5,
    )
    metadata = dict(pack.retrieval_metadata or {})
    # Even with a full 105s budget, the tail cannot grind 8+ rounds:
    # at most 4 semantic model calls are issued per turn.
    assert model.calls <= 4
    assert int(metadata.get("loop_model_calls", 0)) <= 4
    assert int(metadata.get("provider_call_count", 0)) <= 4


async def test_fast_path_still_reaches_ready(monkeypatch: Any) -> None:
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-3:ru:first", text="Synthetic calm routine ALPHA marker nearby.")
    index = _fake_index([first])
    _install_branch(
        monkeypatch,
        [{first.chunk_id: _fused(first.chunk_id, 50.0)}],
        [[first.chunk_id]],
    )

    class _ReadyModel:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            import re as _re

            _ = (system, schema, retry_count)
            if "<passages>" in prompt and "<original_request>" in prompt:
                passage_texts: dict[str, str] = {}
                for match in _re.finditer(
                    r'<passage id="([^"]+)"[^>]*>(.*?)</passage>', prompt, _re.DOTALL
                ):
                    pid, body = match.group(1), match.group(2)
                    passage_texts[pid] = " ".join(body.split())
                supporting = [pid for pid, body in passage_texts.items() if "ALPHA marker" in body]
                if supporting:
                    return {
                        "needs": [
                            {
                                "need_id": "need-1",
                                "covered": True,
                                "supporting_passage_ids": supporting,
                                "supporting_quote": "ALPHA marker",
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
                            "missing": "synthetic shortfall",
                        }
                    ]
                }
            return {
                "selected_chunk_ids": [first.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            }

    pack = await aretrieve_with_semantic_selection(
        index,
        ["synthetic calm routine query"],
        resolved_intent="synthetic calm routine intent",
        conversation_context="",
        user_message="synthetic calm routine request",
        selection_model=_ReadyModel(),
        information_needs=[{"need_id": "need-1", "text": "synthetic calm routine"}],
        max_iterations=3,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "ready"
    assert len(list(pack.passages)) > 0


def test_no_literal_qualification_branches() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "retrieval_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "turn_budget.py").read_text(encoding="utf-8"),
    ]
    for source in sources:
        for fragment in (
            "Чем ты вообще можешь быть полезен",
            "Слушай, а ты тут вообще чем помочь можешь",
            "К вечеру очень тянет выпить",
            "Под вечер опять тянеет выпить",
            "Дома снова ссора",
            "Поругались дома из-за бухла",
            "покупать акции",
            "Какой телефон сейчас выгоднее купить",
            "Привет! К вечеру очень тянет выпить",
            "Привет, подскажи че делать когда под вечер тянеет выпить",
        ):
            assert fragment not in source
    node_source = sources[0]
    for stem in ("тяне", "тянет выпить", "бухл", "похмел"):
        assert stem not in node_source
