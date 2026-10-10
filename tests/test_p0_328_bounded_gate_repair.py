"""P0 bounded repair (kodmial/aa#328): conversational certification + retrieval budget.

- Planner-certified conversational glue (model-resolved conversational
  mode, zero queries, empty pack) certifies as clarification without a
  false book requirement, even for long dialogue replies. Any book
  unit or non-empty pack still takes the strong answer path.
- Retrieval read/coverage loop stops issuing further model-backed
  iterations once the turn deadline is spent and returns a typed
  exhausted outcome (never fake coverage). The 5s local RRF value is
  diagnostic only (kodmial/aa#335).
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack_entry(passage_id: str = "p1") -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": "Поддержка рядом помогает.",
        "source_id": "ru-fourth-edition-txt",
        "section_id": "ru-test/sec-a",
        "child_chunk_ids": ["c1"],
        "char_start": 0,
        "char_end": 10,
        "text_sha256": _sha("Поддержка рядом помогает."),
        "source_sha256": _sha("src"),
        "corpus_version": "ru-v1",
    }


def test_conversational_long_reply_is_clarification_without_book_requirement() -> None:
    from aa.conversation.finalization import (
        AnswerCandidate,
        candidate_from_state,
        certify_candidate,
        context_digest_for_turn,
        infer_outcome_kind,
        normalize_answer_text,
        verify_certificate,
    )

    dialogue = (
        "Помогаю разбирать тягу и ближайшие шаги спокойно и по порядку. "
        "Расскажите, что сейчас беспокоит сильнее всего и что уже пробовали?"
    )
    assert len(dialogue.split()) >= 12
    assert (
        infer_outcome_kind(
            text=dialogue,
            evidence_pack=[],
            grounding_result=None,
            telemetry={"planner_mode": "conversational", "planner_query_count": 0},
        )
        == "clarification"
    )
    question = "Чем ты вообще можешь быть полезен здесь?"
    context_digest_for_turn(
        question=question, resolved_intent=question, summary="", recent_texts=[question]
    )
    state = {
        "current_user_message": question,
        "resolved_intent": "",
        "conversation_summary": "",
        "messages": [HumanMessage(content=question)],
        "evidence_pack": [],
        "final_response": dialogue,
        "draft_response": dialogue,
        "grounding_result": None,
        "retry_state": {"planner_mode": "conversational", "planner_query_count": 0},
        "search_queries": [],
        "planner_mode": "conversational",
        "response_unit_texts": [],
        "route": "normal",
    }
    candidate, grounding, q, intent, summary, messages, texts = candidate_from_state(state)
    assert candidate.outcome_kind == "clarification"
    certificate = certify_candidate(
        candidate=candidate,
        grounding_result=grounding,
        question=q,
        resolved_intent=intent,
        summary=summary,
        recent_messages=messages,
        stored_unit_texts=texts,
    )
    verify_certificate(candidate=candidate, certificate=certificate)
    _ = AnswerCandidate
    _ = normalize_answer_text


def test_substantive_long_reply_without_provenance_stays_answer_and_fails_closed() -> None:
    from aa.conversation.finalization import (
        FinalizationError,
        candidate_from_state,
        certify_candidate,
    )

    question = "К вечеру очень тянет выпить, как с этим обходиться?"
    long_no_evidence = (
        "Тяга вечером тяжело переживается, поддержка рядом помогает обсудить "
        "ближайший шаг и обратиться за помощью вовремя сегодня."
    )
    assert len(long_no_evidence.split()) >= 12
    state = {
        "current_user_message": question,
        "resolved_intent": question,
        "conversation_summary": "",
        "messages": [HumanMessage(content=question)],
        "evidence_pack": [],
        "final_response": long_no_evidence,
        "draft_response": long_no_evidence,
        "grounding_result": None,
        "retry_state": {"planner_mode": "retrieval", "planner_query_count": 3},
        "search_queries": ["q1", "q2", "q3"],
        "planner_mode": "retrieval",
        "response_unit_texts": [],
        "route": "normal",
    }
    candidate, grounding, q, intent, summary, messages, texts = candidate_from_state(state)
    assert candidate.outcome_kind == "answer"
    try:
        certify_candidate(
            candidate=candidate,
            grounding_result=grounding,
            question=q,
            resolved_intent=intent,
            summary=summary,
            recent_messages=messages,
            stored_unit_texts=texts,
        )
    except FinalizationError as exc:
        assert exc.category in ("claim-missing", "missing-verdicts", "unsupported-claims")
    else:
        raise AssertionError("substantive bookless answer must fail closed")


def test_conversational_claim_with_book_evidence_stays_answer() -> None:
    from aa.conversation.finalization import infer_outcome_kind

    assert (
        infer_outcome_kind(
            text="Коротко.",
            evidence_pack=[_pack_entry()],
            grounding_result=None,
            telemetry={"planner_mode": "conversational", "planner_query_count": 0},
        )
        == "answer"
    )
    assert (
        infer_outcome_kind(
            text="Коротко.",
            evidence_pack=[],
            grounding_result={"units": [{"scope": "book"}], "all_required_supported": False},
            telemetry={"planner_mode": "conversational", "planner_query_count": 0},
        )
        == "answer"
    )


def _record(
    chunk_id: str,
    *,
    section: str = "chapter-1",
    start: int = 0,
    text: str = "Текст про поддержку и трезвость.",
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


class _ScriptedLoopModel:
    """Serve scripted selection answers; coverage judges by marker presence."""

    def __init__(self, selections: list[dict[str, Any]], markers: dict[str, str]) -> None:
        self._selections = list(selections)
        self._markers = dict(markers)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        import re as _re

        _ = (system, schema, retry_count)
        self.calls += 1
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


async def test_retrieval_loop_stops_after_interactive_budget_with_typed_exhaustion(
    monkeypatch: Any,
) -> None:
    import aa.retrieval.evidence as evidence_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection
    from aa.conversation.turn_budget import new_turn_budget

    first = _record("chapter-1:ru:first", start=0, text="Начало про утренний разбор.")
    second = _record("chapter-2:ru:second", start=0, text="Продолжение без ответа.")
    index = _fake_index([first, second])
    pools = [
        {first.chunk_id: _fused(first.chunk_id, 50.0)},
        {second.chunk_id: _fused(second.chunk_id, 60.0)},
    ]
    per_ids = [[first.chunk_id], [second.chunk_id]]
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
    # Every iteration requests more detail, so an unbounded loop would
    # run all three iterations; the marker is unreachable so coverage
    # never closes on its own.
    model = _ScriptedLoopModel(
        selections=[
            {
                "selected_chunk_ids": [first.chunk_id],
                "need_more_detail": True,
                "followup_queries": ["другой запрос про поддержку"],
            },
            {
                "selected_chunk_ids": [second.chunk_id],
                "need_more_detail": True,
                "followup_queries": ["еще запрос про поддержку"],
            },
            {
                "selected_chunk_ids": [second.chunk_id],
                "need_more_detail": True,
                "followup_queries": ["еще один запрос"],
            },
        ],
        markers={"need-1": "НЕДОСТИЖИМЫЙ-МАРКЕР"},
    )
    # kodmial/aa#335: the turn deadline (not the 5s local RRF
    # diagnostic) bounds further iterations. A nearly-spent turn budget
    # stops the loop with typed exhaustion instead of grinding.
    _ = monkeypatch
    _ = evidence_mod
    tight_budget = new_turn_budget(deadline_s=0.12)
    try:
        pack = await aretrieve_with_semantic_selection(
            index,
            ["утренний разбор тяги"],
            resolved_intent="утренний разбор тяги",
            conversation_context="",
            user_message="как разбирать тягу утром",
            selection_model=model,
            information_needs=[{"need_id": "need-1", "text": "утренний разбор"}],
            max_iterations=3,
            turn_budget=tight_budget,
        )
    finally:
        pass
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "exhausted"
    assert metadata.get("coverage_exhaustion_reason") == "latency-budget"
    assert metadata.get("coverage_all_covered") is False
    # First discover/select/read/assess pair preserved (model-driven
    # coverage ran once); no grinding across three iterations.
    assert int(metadata.get("loop_model_calls", 0)) <= 2
    assert int(metadata.get("loop_iterations", 0)) <= 1
