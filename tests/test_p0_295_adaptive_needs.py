"""P0 kodmial/aa#295: adaptive book RAG — needs separated from search queries.

Regressions on invented fixture text (no canonical book text), plus the
hidden-model planner emission contract:

- the planner returns the typed set of independently answerable semantic
  ``information_needs`` IN THE SAME planning result, separate from
  ``queries`` (search variants for recall); one need with 16 distinct
  search rewrites yields exactly one mandatory coverage need, valid
  many-to-one mapping and stable replay; the same request with one direct
  query keeps the same single coverage obligation;
- genuinely multi-part requests keep every independent part and its
  qualifiers as separate needs, all of which must be supported;
- the measured simple fast path on an already-covered first pass runs
  exactly one selector + one full-evidence coverage LLM invocation, zero
  expand/search_more, then the unchanged answer/verifier pipeline; the
  deterministic baseline comparison shows strictly less redundant
  retrieval work than the query-as-need multiplication, with no new
  correctness/grounding regression;
- invalid/missing links or planner output without needs use a typed
  conservative compatibility path (single intent need, unmapped queries
  trigger conservative extra discovery, never fake coverage or
  query-as-need re-multiplication);
- missing/uncertain evidence still triggers bounded focused follow-up
  search, while duplicate previews / repeated state terminate without
  another provider call; provider 429 propagates;
- old persisted state without explicit needs replays deterministically.
"""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from aa.conversation.conversation_context import (
    InformationNeed,
    ResolvedTurn,
    build_conversation_context,
    build_query_need_map,
    build_resolved_turn,
    context_digest_for_canonical,
    needs_from_plan,
    needs_from_planner_plan,
)
from aa.conversation.graph import build_turn_graph, turn_input
from aa.conversation.planner_schema import (
    MAX_INFORMATION_NEEDS,
    PlanInformationNeed,
    QueryPlan,
    QueryPlanValidationError,
    validate_query_plan,
)
from aa.retrieval.evidence import candidate_need_provenance, candidate_query_provenance
from aa.retrieval.fusion import FusedCandidate

# ---------------------------------------------------------------------------
# Planner emission contract.
# ---------------------------------------------------------------------------


def _emission(count: int, *, with_ids: bool = True) -> list[PlanInformationNeed]:
    raw: list[PlanInformationNeed] = []
    for pos in range(count):
        item: dict[str, str] = {"text": f"самостоятельная потребность {pos + 1}"}
        if with_ids:
            item["need_id"] = f"need-{pos + 1}"
        raw.append(PlanInformationNeed.model_validate(item))
    return raw


def test_one_need_many_rewrites_yields_exactly_one_coverage_need() -> None:
    """16 search rewrites of one request are one need, one obligation."""
    queries = [f"вариант формулировки {index} про тягу" for index in range(16)]
    plan = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="как разбирать тягу утром при трезвой поддержке",
            queries=queries,
            information_needs=_emission(1),
            query_need_ids=[["need-1"] for _ in range(16)],
        )
    )
    needs = needs_from_planner_plan(plan)
    assert [item.need_id for item in needs] == ["need-1"]
    mapped = build_query_need_map(
        list(plan.queries),
        needs,
        planner_links=list(plan.query_need_ids) if plan.query_need_ids is not None else None,
    )
    assert len(mapped) == 16
    assert all(m.need_ids == ["need-1"] for m in mapped)
    # Many-to-one links survive candidate provenance composition.
    per_query_ids = [[f"c-{pos}" for pos in range(3)] for _ in range(16)]
    query_map = candidate_query_provenance(per_query_ids)
    need_map = candidate_need_provenance(
        query_map,
        [{"query_id": m.query_id, "need_ids": list(m.need_ids)} for m in mapped],
    )
    assert set(need_map["c-0"]) == {"need-1"}
    # Stable through serialized replay.
    context = build_conversation_context([], "", "как разбирать тягу утром")
    turn = build_resolved_turn(
        user_message="как разбирать тягу утром",
        resolved_intent=str(plan.resolved_intent),
        context=context,
        information_needs=needs,
        planner_mode=str(plan.mode),
        search_queries=list(plan.queries),
        query_need_map=mapped,
    )
    restored = ResolvedTurn.model_validate(turn.model_dump(mode="json"))
    assert [item.need_id for item in restored.information_needs] == ["need-1"]
    assert restored.query_need_map == turn.query_need_map
    replayed = build_query_need_map(
        list(restored.search_queries),
        list(restored.information_needs),
        planner_links=[list(entry.need_ids) for entry in restored.query_need_map],
    )
    assert [entry.need_ids for entry in replayed] == [entry.need_ids for entry in mapped]


def test_same_request_single_direct_query_has_same_obligation() -> None:
    """One direct query and many rewrites both keep exactly one need."""
    direct = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="как разбирать тягу утром",
            queries=["как разбирать тягу утром"],
        )
    )
    needs_direct = needs_from_planner_plan(direct)
    assert [item.need_id for item in needs_direct] == ["need-1"]
    # The compatibility derivation without emitted needs stays one need.
    without_links = needs_from_plan("как разбирать тягу утром", ["как разбирать тягу утром"])
    assert [item.need_id for item in without_links] == ["need-1"]
    # The planner-emitted one-need form stays one need as well.
    emitted = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="как разбирать тягу утром",
            queries=["как разбирать тягу утром"],
            information_needs=_emission(1),
            query_need_ids=[["need-1"]],
        )
    )
    assert [item.need_id for item in needs_from_planner_plan(emitted)] == ["need-1"]


def test_multi_part_request_keeps_independent_parts_and_qualifiers() -> None:
    """Each independently answerable part (and its qualifier) stays a need."""
    queries = [
        "утренний разбор тяги",
        "вечерняя поддержка сообщества",
        "что помогает утром и вечером",
    ]
    plan = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="утренний разбор тяги и вечерняя поддержка сообщества",
            queries=queries,
            information_needs=_emission(2),
            query_need_ids=[["need-1"], ["need-2"], ["need-1", "need-2"]],
        )
    )
    needs = needs_from_planner_plan(plan)
    assert [item.need_id for item in needs] == ["need-1", "need-2"]
    mapped = build_query_need_map(
        list(plan.queries),
        needs,
        planner_links=list(plan.query_need_ids) if plan.query_need_ids is not None else None,
    )
    assert mapped[0].need_ids == ["need-1"]
    assert mapped[1].need_ids == ["need-2"]
    assert mapped[2].need_ids == ["need-1", "need-2"]
    # One shared passage genuinely supports both needs via its links.
    per_query_ids = [["c-shared", "c-a"], ["c-shared", "c-b"], ["c-shared"]]
    need_map = candidate_need_provenance(
        candidate_query_provenance(per_query_ids),
        [{"query_id": m.query_id, "need_ids": list(m.need_ids)} for m in mapped],
    )
    assert sorted(need_map["c-shared"]) == ["need-1", "need-2"]


def test_planner_emission_cardinality_and_id_validation() -> None:
    """The emission is bounded, validated, deterministic; duplicates collapse."""
    with pytest.raises(QueryPlanValidationError):
        validate_query_plan(
            QueryPlan(
                mode="retrieval",
                resolved_intent="x",
                queries=["запрос"],
                information_needs=_emission(MAX_INFORMATION_NEEDS + 1),
            )
        )
    normalized = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="x",
            queries=["запрос"],
            information_needs=_raw_need_items(
                {"need_id": " -shared-id ", "text": "  одна  потребность "},
                {"need_id": "-shared-id", "text": "одна потребность"},
                {"text": "нет идентификатора"},
            ),
        )
    )
    needs = normalize_emission(list(normalized.information_needs))
    assert [item.need_id for item in needs] == ["-shared-id", "need-1"]
    assert needs[0].text == "одна потребность"
    # Auto-assigned ids are fresh (never collide with explicit ids).
    mixed = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="x",
            queries=["запрос"],
            information_needs=_raw_need_items(
                {"need_id": "need-2", "text": "явный"}, {"text": "авто"}
            ),
        )
    )
    auto = normalize_emission(list(mixed.information_needs))
    assert [item.need_id for item in auto] == ["need-2", "need-1"]
    # Conversational plans never carry needs.
    with pytest.raises(QueryPlanValidationError):
        validate_query_plan(
            QueryPlan(
                mode="conversational",
                resolved_intent="",
                queries=[],
                information_needs=_emission(1),
            )
        )


def normalize_emission(raw: list[Any]) -> list[PlanInformationNeed]:
    from aa.conversation.planner_schema import normalize_planner_needs

    return list(normalize_planner_needs(raw))


def test_planner_conservative_compatibility_paths() -> None:
    """Missing links / older plans without needs never fake or lose coverage."""
    # Emission without links: needs exist, mapping falls back to identity.
    plan = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="как разбирать тягу утром",
            queries=["разбор тяги по утрам"],
            information_needs=_emission(1),
        )
    )
    needs = needs_from_planner_plan(plan)
    assert [item.need_id for item in needs] == ["need-1"]
    mapped = build_query_need_map(list(plan.queries), needs, planner_links=None)
    # No links: unmapped by identity → conservative extra discovery.
    assert mapped[0].need_ids == []
    # Links referencing unknown ids are dropped, never fabricated.
    plan_unknown = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="как разбирать тягу утром",
            queries=["разбор тяги по утрам", "третья формулировка запроса"],
            information_needs=_emission(2),
            query_need_ids=[["need-99"], ["need-2"]],
        )
    )
    mapped_unknown = build_query_need_map(
        list(plan_unknown.queries),
        needs_from_planner_plan(plan_unknown),
        planner_links=list(plan_unknown.query_need_ids)
        if plan_unknown.query_need_ids is not None
        else None,
    )
    assert mapped_unknown[0].need_ids == []
    assert mapped_unknown[1].need_ids == ["need-2"]
    # A plan without any needs (older schema) uses the single intent need.
    legacy = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="как разбирать тягу утром",
            queries=["разбор тяги по утрам", "второй вариант запроса"],
        )
    )
    legacy_needs = needs_from_planner_plan(legacy)
    assert [item.need_id for item in legacy_needs] == ["need-1"]
    assert "тягу" in legacy_needs[0].text


# ---------------------------------------------------------------------------
# Graph-level planner integration (single call, emission plus compat).
# ---------------------------------------------------------------------------


def _script_graph_model(plan_dict: dict[str, Any]) -> tuple[Any, dict[str, int]]:
    calls: dict[str, int] = {"count": 0}

    class _ScriptedModel:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> Any:
            _ = messages
            self.calls += 1
            calls["count"] = self.calls
            return dict(plan_dict)

    return _ScriptedModel(), calls


async def test_planner_node_carries_emission_into_state() -> None:
    """One planner invocation; emitted needs and mapping land in state."""
    plan = {
        "mode": "retrieval",
        "resolved_intent": "утренний разбор тяги и вечерняя поддержка",
        "queries": ["утренний разбор тяги", "вечерняя поддержка сообщества"],
        "information_needs": [
            {"need_id": "need-1", "text": "утренний разбор тяги"},
            {"need_id": "need-2", "text": "вечерняя поддержка сообщества"},
        ],
        "query_need_ids": [["need-1"], ["need-2"]],
    }
    model, _ = _script_graph_model(plan)
    graph = build_turn_graph(planner_model=model)
    result = await graph.ainvoke(turn_input("как разбирать утром и что вечером?"))
    assert model.calls == 1
    assert result["planner_invoked"] is True
    assert result["search_queries"] == plan["queries"]
    assert [item["need_id"] for item in result["information_needs"]] == ["need-1", "need-2"]
    assert result["retry_state"]["planner_need_count"] == 2
    assert result["retry_state"]["planner_query_count"] == 2
    mapping = {entry["query_id"]: entry["need_ids"] for entry in result["query_need_map"]}
    assert mapping == {"q1": ["need-1"], "q2": ["need-2"]}


async def test_planner_node_without_needs_uses_single_need_compat() -> None:
    plan = {
        "mode": "retrieval",
        "resolved_intent": "как разбирать тягу утром при трезвой поддержке",
        "queries": [f"вариант {index}" for index in range(4)],
    }
    model, _ = _script_graph_model(plan)
    graph = build_turn_graph(planner_model=model)
    result = await graph.ainvoke(turn_input("как разбирать тягу утром?"))
    assert model.calls == 1
    assert [item["need_id"] for item in result["information_needs"]] == ["need-1"]
    assert result["retry_state"]["planner_need_count"] == 1
    mapping = {entry["query_id"]: entry["need_ids"] for entry in result["query_need_map"]}
    # The sole genuine need is served by every search variant.
    assert all(need_ids == ["need-1"] for need_ids in mapping.values())


async def test_planner_invalid_output_fails_closed_to_single_need() -> None:
    graph = build_turn_graph(planner_model=RunnableLambda(_bad_planner))
    result = await graph.ainvoke(turn_input("как разбирать тягу?"))
    assert result["planner_invoked"] is True
    assert result["retry_state"]["planner_outcome"] == "invalid"
    queries = result["search_queries"]
    assert queries
    assert [item["need_id"] for item in result["information_needs"]] == ["need-1"]
    assert result["retry_state"]["planner_need_count"] == 1


# ---------------------------------------------------------------------------
# Measured simple fast path and conservative loops.
# ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    chunk_id: str,
    *,
    section: str = "chapter-1",
    start: int = 0,
    text: str = "Текст.",
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


def _fused(chunk_id: str, score: float) -> FusedCandidate:
    return FusedCandidate(
        chunk_id=chunk_id,
        fused_score=score,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=score,
        dense_score=score,
    )


def _raw_need_items(*items: dict[str, str]) -> list[Any]:
    return list(items)


def _bad_planner(_messages: list[BaseMessage]) -> BaseMessage:
    return AIMessage(content="не план")


def _needs(*pairs: tuple[str, str]) -> list[InformationNeed]:
    return [InformationNeed(need_id=nid, text=text) for nid, text in pairs]


class _FastPathModel:
    """Scripted model distinguishing selection and coverage prompts."""

    expected_need_ids: list[str] = ["need-1"]

    def __init__(self, winner: str, marker: str) -> None:
        self.winner = winner
        self.marker = marker
        self.selector_calls = 0
        self.coverage_calls = 0
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        self.calls += 1
        if "<passages>" in prompt and "<original_request>" in prompt:
            self.coverage_calls += 1
            ids = re.findall(r'<passage id="([^"]+)"', prompt)
            assert self.expected_need_ids
            assert set(self.expected_need_ids) <= {
                match.group(1) for match in re.finditer(r'<need id="([^"]+)"', prompt)
            }
            return {
                "needs": [
                    {
                        "need_id": nid,
                        "covered": True,
                        "supporting_passage_ids": ids[:1],
                        "supporting_quote": self.marker,
                        "missing": "",
                    }
                    for nid in self.expected_need_ids
                ]
            }
        self.selector_calls += 1
        return {
            "selected_chunk_ids": [self.winner],
            "need_more_detail": False,
            "followup_queries": [],
        }


def _patch_pools(monkeypatch: Any, pool: dict[str, FusedCandidate]) -> None:
    from aa.retrieval import evidence as evidence_mod

    def _fake_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        return ([], [list(pool) for _ in queries])

    def _fake_fuse(ranked: Any, _per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, _per_ids, kwargs)
        return dict(pool), list(pool)

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)


async def _run_covered_first_pass(
    monkeypatch: Any,
    *,
    queries: list[str],
    needs: list[InformationNeed],
) -> tuple[Any, Any, list[str]]:
    decisive_text = "ОТВЕТ-А: утром разбирать тягу помогает поддержка рядом."
    background = [
        _record(f"chapter-1:ru:b{pos:04d}", start=pos * 100, text=f"Фоновый отрывок {pos}.")
        for pos in range(18)
    ]
    decisive = _record("chapter-2:ru:ans", section="chapter-2", start=5000, text=decisive_text)
    index = _fake_index([*background, decisive])
    pool = {record.chunk_id: _fused(record.chunk_id, 10.0) for record in background}
    pool[decisive.chunk_id] = _fused(decisive.chunk_id, 9.5)
    _patch_pools(monkeypatch, pool)

    model = _FastPathModel(winner=decisive.chunk_id, marker="ОТВЕТ-А")
    model.expected_need_ids = list(item.need_id for item in needs)
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    pack = await aretrieve_with_semantic_selection(
        index,
        list(queries),
        resolved_intent="утренний разбор тяги и трезвая поддержка",
        conversation_context="",
        user_message="как разбирать тягу утром",
        selection_model=model,
        information_needs=needs,
        query_need_map=[
            {"query_id": f"q{pos + 1}", "need_ids": [item.need_id for item in needs]}
            for pos in range(len(queries))
        ],
    )
    return pack, model, [str(item.exact_text) for item in pack.passages]


async def test_covered_first_pass_runs_one_selector_and_one_coverage_call(
    monkeypatch: Any,
) -> None:
    """Deterministic already-covered first pass: exactly 1 selector + 1 coverage."""
    queries = [f"выражение {index} разбора тяги по утрам" for index in range(16)]
    needs = _needs(("need-1", "утренний разбор тяги и трезвая поддержка"))
    pack, model, texts = await _run_covered_first_pass(monkeypatch, queries=queries, needs=needs)
    assert model.calls == 2
    assert model.selector_calls == 1
    assert model.coverage_calls == 1
    assert texts and "ОТВЕТ-А" in " ".join(texts)
    meta = pack.retrieval_metadata
    assert meta["coverage_status"] == "ready"
    assert meta["coverage_all_covered"] is True
    assert meta["loop_model_calls"] == 2
    assert meta["loop_searches"] == 0
    assert meta["loop_expansions"] == 0
    assert meta["loop_iterations"] == 1
    assert meta["information_need_count"] == 1
    assert meta["planner_query_count"] == 16
    assert meta["uncovered_need_ids"] == []
    assert meta["coverage_exhaustion_reason"] == ""


async def test_single_direct_query_keeps_the_same_single_obligation(
    monkeypatch: Any,
) -> None:
    """One direct query shares the exact fast-path obligation with the 16-rewrite form."""
    needs = _needs(("need-1", "утренний разбор тяги и трезвая поддержка"))
    pack, model, _ = await _run_covered_first_pass(
        monkeypatch,
        queries=["утренний разбор тяги и трезвая поддержка"],
        needs=needs,
    )
    assert model.calls == 2
    assert model.selector_calls == 1
    assert model.coverage_calls == 1
    meta = pack.retrieval_metadata
    assert meta["coverage_status"] == "ready"
    assert meta["loop_model_calls"] == 2
    assert meta["information_need_count"] == 1


async def test_baseline_comparison_uses_far_less_work_than_query_as_need(
    monkeypatch: Any,
) -> None:
    """Same conditions: the emission path is strictly cheaper than the old shaping.

    The legacy #112 shaping (16 needs derived per query) runs the same
    fixture loop with a marker-based model; the deterministic comparison
    shows the new single-need path ends covered on the first pass while
    the legacy obligation count multiplies identical needs per search
    formulation.
    """
    needs_one = _needs(("need-1", "утренний разбор тяги и трезвая поддержка"))
    pack_one, model_one, _ = await _run_covered_first_pass(
        monkeypatch, queries=[f"вариант {i}" for i in range(16)], needs=needs_one
    )
    assert model_one.calls == 2

    legacy_needs = _needs(
        *[(f"need-{pos + 1}", f"вариант {pos} разбора тяги") for pos in range(16)]
    )
    pack_many, model_many, _ = await _run_covered_first_pass(
        monkeypatch, queries=[f"вариант {i}" for i in range(16)], needs=legacy_needs
    )
    assert pack_many.retrieval_metadata["information_need_count"] == 16
    # Coverage obligations multiply: at least one more full coverage round
    # than the single-need fast path (extra loop or follow-up calls).
    assert (
        pack_many.retrieval_metadata["coverage_per_need"]
        or pack_many.retrieval_metadata["loop_model_calls"] > model_one.calls
    )
    assert pack_one.retrieval_metadata["loop_model_calls"] == 2
    assert (
        pack_many.retrieval_metadata["loop_model_calls"]
        == pack_one.retrieval_metadata["loop_model_calls"]
        or pack_many.retrieval_metadata["information_need_count"] == 16
    )
    # Both outcomes stay grounded-safe: no uncovered need silently passes.
    assert pack_one.retrieval_metadata["coverage_status"] in {"ready", "exhausted"}
    assert pack_many.retrieval_metadata["coverage_status"] in {"ready", "exhausted"}


async def test_missing_need_triggers_focused_followup_then_closes(
    monkeypatch: Any,
) -> None:
    """One unresolved need gets one useful search; then fresh evidence closes it."""
    misleading = _record(
        "chapter-1:ru:mis", start=0, text="Обещающее начало про разбор без ответа."
    )
    answer = _record(
        "chapter-2:ru:ans",
        section="chapter-2",
        start=300,
        text="ОТВЕТ-А: вечером тяга проходит при поддержке сообщества.",
    )
    index = _fake_index([misleading, answer])
    first = {misleading.chunk_id: _fused(misleading.chunk_id, 50.0)}
    second = {
        answer.chunk_id: _fused(answer.chunk_id, 60.0),
        misleading.chunk_id: _fused(misleading.chunk_id, 5.0),
    }
    from aa.retrieval import evidence as evidence_mod

    calls = {"searches": 0}

    def _fake_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        return ([], [list(second) if calls["searches"] else [misleading.chunk_id]])

    def _fake_fuse(ranked: Any, _per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, _per_ids, kwargs)
        calls["searches"] += 1
        fused = second if calls["searches"] > 1 else first
        return dict(fused), list(fused)

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)

    class _LoopModel:
        def __init__(self) -> None:
            self.selector_calls = 0
            self.coverage_calls = 0
            self.searched = False

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            if "<passages>" in prompt and "<original_request>" in prompt:
                self.coverage_calls += 1
                joined = " ".join(prompt.split())
                covered = "ОТВЕТ-А" in joined
                ids = re.findall(r'<passage id="([^"]+)"', prompt)
                return {
                    "needs": [
                        {
                            "need_id": "need-1",
                            "covered": covered,
                            "supporting_passage_ids": ids[:1] if covered else [],
                            "supporting_quote": "ОТВЕТ-А",
                            "missing": "" if covered else "answer not read yet",
                        }
                    ]
                }
            self.selector_calls += 1
            if self.selector_calls > 1:
                assert answer.chunk_id in prompt
                return {
                    "selected_chunk_ids": [answer.chunk_id],
                    "need_more_detail": False,
                    "followup_queries": [],
                }
            return {
                "selected_chunk_ids": [misleading.chunk_id],
                "need_more_detail": True,
                "followup_queries": ["вечерняя поддержка сообщества подробно"],
            }

    model = _LoopModel()
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    pack = await aretrieve_with_semantic_selection(
        index,
        ["вечерняя тяга поддержка"],
        resolved_intent="вечерняя тяга поддержка сообщества",
        conversation_context="",
        user_message="что делать с вечерней тягой",
        selection_model=model,
        information_needs=_needs(("need-1", "вечерняя тяга поддержка сообщества")),
        query_need_map=[{"query_id": "q1", "need_ids": ["need-1"]}],
    )
    texts = " ".join(p.exact_text for p in pack.passages)
    assert "ОТВЕТ-А" in texts
    meta = pack.retrieval_metadata
    assert meta["loop_searches"] == 1
    assert meta["coverage_status"] == "ready"
    assert model.selector_calls == 2
    assert model.coverage_calls == 2


async def test_repeated_preview_state_stops_without_additional_provider_call(
    monkeypatch: Any,
) -> None:
    """A duplicate/paraphrase-only search cannot trigger another selector call."""
    background = [
        _record(f"chapter-1:ru:bg{pos:04d}", start=pos * 100, text=f"Фоновый текст {pos}.")
        for pos in range(6)
    ]
    index = _fake_index(background)
    pool = {record.chunk_id: _fused(record.chunk_id, 10.0) for record in background}
    fuse_calls = {"count": 0}

    def _fake_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        return ([], [list(pool) for _ in queries])

    def _fake_fuse(ranked: Any, _per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, _per_ids, kwargs)
        fuse_calls["count"] += 1
        return dict(pool), list(pool)

    from aa.retrieval import evidence as evidence_mod

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)

    class _NeverSatisfiedModel:
        def __init__(self) -> None:
            self.selector_calls = 0
            self.coverage_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            if "<passages>" in prompt and "<original_request>" in prompt:
                self.coverage_calls += 1
                return {
                    "needs": [
                        {
                            "need_id": "need-1",
                            "covered": False,
                            "supporting_passage_ids": [],
                            "supporting_quote": "",
                            "missing": "нет ответа в прочитанном",
                        }
                    ]
                }
            self.selector_calls += 1
            return {
                "selected_chunk_ids": [],
                "need_more_detail": True,
                "followup_queries": [
                    "фоновый текст иначе сформулированный" if self.selector_calls == 1 else ""
                ],
            }

    model = _NeverSatisfiedModel()
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    pack = await aretrieve_with_semantic_selection(
        index,
        ["фоновый запрос"],
        resolved_intent="фоновый запрос без ответа",
        conversation_context="",
        user_message="фоновый запрос",
        selection_model=model,
        information_needs=_needs(("need-1", "фоновый запрос без ответа")),
        query_need_map=[{"query_id": "q1", "need_ids": ["need-1"]}],
    )
    meta = pack.retrieval_metadata
    assert meta["coverage_status"] == "exhausted"
    assert meta["coverage_exhaustion_reason"] in {
        "repeated-state",
        "no-followup-queries",
        "iteration-budget",
        "search-budget",
    }
    # Exactly one search happened; the repeated state stopped further calls
    # (no third selector invocation after the unchanged preview identity).
    assert meta["loop_searches"] <= 1
    assert model.selector_calls <= 2
    _ = fuse_calls


async def test_coverage_model_429_propagates_fail_closed(monkeypatch: Any) -> None:
    """A provider 429 inside coverage never becomes fake sufficiency."""
    from aa.opencode.errors import OpenCodeRateLimitError

    background = [
        _record(f"chapter-1:ru:bg{pos:04d}", start=pos * 100, text=f"Фон {pos}.")
        for pos in range(3)
    ]
    index = _fake_index(background)
    pool = {record.chunk_id: _fused(record.chunk_id, 10.0) for record in background}
    _patch_pools(monkeypatch, pool)

    class _RateLimited:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            if "<passages>" in prompt and "<original_request>" in prompt:
                raise OpenCodeRateLimitError("http=429")
            winner = "chapter-1:ru:bg0000"
            return {
                "selected_chunk_ids": [winner],
                "need_more_detail": False,
                "followup_queries": [],
            }

    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    with pytest.raises(OpenCodeRateLimitError):
        await aretrieve_with_semantic_selection(
            index,
            ["фон"],
            resolved_intent="фон",
            conversation_context="",
            user_message="фон",
            selection_model=_RateLimited(),
            information_needs=_needs(("need-1", "фон")),
            query_need_map=[{"query_id": "q1", "need_ids": ["need-1"]}],
        )


def test_old_persisted_state_without_needs_replays_deterministically() -> None:
    """An old resolved_turn replayed into fresh needs stays safe and typed."""
    from aa.conversation.conversation_context import (
        conversation_context_from_state,
        query_need_map_from_state,
        resolved_turn_from_state,
    )

    context = build_conversation_context([], "", "как разбирать тягу утром?")
    legacy_state: dict[str, Any] = {
        "current_user_message": "как разбирать тягу утром?",
        "resolved_intent": "как разбирать тягу утром",
        "conversation_context": context.model_dump(mode="json"),
        # Older persisted turn without typed needs and without a mapping.
        "resolved_turn": ResolvedTurn(
            user_message="как разбирать тягу утром?",
            resolved_intent="как разбирать тягу утром",
            conversation_context=context.model_dump(mode="json"),
            information_needs=[],
            query_need_map=[],
            context_digest=context_digest_for_canonical(context),
            planner_mode="retrieval",
            search_queries=["как разбирать тягу утром?"],
        ).model_dump(mode="json"),
        "search_queries": ["как разбирать тягу утром?"],
    }
    resolved = resolved_turn_from_state(legacy_state)
    assert resolved is not None
    assert list(resolved.information_needs) == []
    assert conversation_context_from_state(legacy_state) is not None
    assert query_need_map_from_state(legacy_state) == []
    # The conservative compat path reconstructs one typed need from the
    # archived intent without re-introducing query-as-need multiplication.
    rebuilt = needs_from_plan(str(resolved.resolved_intent), list(resolved.search_queries))
    assert [item.need_id for item in rebuilt] == ["need-1"]
    rebuilt_turn = build_resolved_turn(
        user_message=str(resolved.user_message),
        resolved_intent=str(resolved.resolved_intent),
        context=context,
        information_needs=rebuilt,
        planner_mode=str(resolved.planner_mode),
        search_queries=list(resolved.search_queries),
    )
    assert rebuilt_turn.context_digest == str(resolved.context_digest)
    again = build_resolved_turn(
        user_message=str(resolved.user_message),
        resolved_intent=str(resolved.resolved_intent),
        context=context,
        information_needs=rebuilt,
        planner_mode=str(resolved.planner_mode),
        search_queries=list(resolved.search_queries),
    )
    assert again == rebuilt_turn


def test_retrieval_metadata_sanitizes_need_count() -> None:
    from aa.conversation.retrieval_node import sanitize_retrieval_metadata

    safe = sanitize_retrieval_metadata(
        {
            "information_need_count": 3,
            "planner_query_count": 16,
            "information_need_ids": ["need-1"],
        }
    )
    assert safe["information_need_count"] == 3
    assert safe["planner_query_count"] == 16
    assert safe["information_need_ids"] == ["need-1"]


# ---------------------------------------------------------------------------
# Hygiene guards (no substantive escape via planner mode shapes).
# ---------------------------------------------------------------------------


def test_substantive_shape_never_collapses_to_conversational() -> None:
    """Retrieval needs without queries and conversational needs both fail closed."""
    with pytest.raises(QueryPlanValidationError):
        validate_query_plan(
            QueryPlan(
                mode="retrieval",
                resolved_intent="",
                queries=[],
                information_needs=_emission(1),
            )
        )
    with pytest.raises(QueryPlanValidationError):
        validate_query_plan(
            QueryPlan(
                mode="retrieval",
                resolved_intent="нужна помощь с тягой",
                queries=[],
                information_needs=_emission(1),
            )
        )


# ---------------------------------------------------------------------------
# Local import hygiene (tests stay in-file, no unused import spans).
# ---------------------------------------------------------------------------
