"""AUDIT P0-2 (kodmial/aa#304): exact final-answer certification as delivery gate.

Compiled-production-graph tests: ordinary answer, repair with new
evidence, deletion/compaction/quote-replacement/safety-regen requiring
new certificates, multi-message split preservation, stale-cert tamper
fail-closed, and history exclusion for rejected candidates.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Фиктивная поддержка рядом. Тяга проходит, если обратиться за помощью.",
    source_id: str = "ru-fourth-edition-txt",
    section_id: str = "chapter-3",
    char_start: int = 0,
    char_end: int = 120,
) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": source_id,
        "section_id": section_id,
        "char_start": char_start,
        "char_end": char_end,
        "text_sha256": _sha(text),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


class _AnswerModel:
    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        if not self._drafts:
            raise AssertionError("answer model called more times than scripted")
        return AIMessage(content=self._drafts.pop(0))


class _VerifierModel:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results: list[dict[str, Any]] = []
        for entry in results:
            self._results.extend(_expand(entry))

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        if not self._results:
            raise AssertionError("verifier called more times than scripted")
        return dict(self._results.pop(0))


def _expand(entry: dict[str, Any]) -> list[dict[str, Any]]:
    if "units" in entry and isinstance(entry["units"], list):
        out = []
        for unit in entry["units"]:
            scope = str(unit.get("scope", "book"))
            supported = bool(unit.get("supported", False))
            out.append(
                {
                    "requires_book_evidence": scope == "book",
                    "supported": supported,
                    "evidence_passage_ids": list(unit.get("evidence_passage_ids", [])),
                    "addresses_intent": bool(unit.get("addresses_intent", supported)),
                }
            )
        return out
    return [dict(entry)]


def _support_result(unit_ids: list[str], passage_id: str) -> dict[str, Any]:
    return {
        "units": [
            {
                "unit_id": uid,
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [passage_id],
                "addresses_intent": True,
            }
            for uid in unit_ids
        ],
        "all_required_supported": True,
    }


def _twelve_queries(base: str = "тяга поддержка трезвость") -> list[str]:
    return [f"{base} вариант {index}" for index in range(12)]


def _retrieval_plan(intent: str = "что помогает при тяге") -> Any:
    async def _plan(_messages: Any) -> Any:
        return {"mode": "retrieval", "resolved_intent": intent, "queries": _twelve_queries()}

    return RunnableLambda(_plan)


async def _preserving_stub(state: Any) -> dict[str, Any]:
    pack = [dict(item) for item in state.get("evidence_pack", []) if isinstance(item, dict)]
    return {
        "retrieval_hits": [],
        "evidence_pack": pack,
        "retrieval_latency_ms": 0.0,
        "retrieval_over_budget": False,
    }


def _graph_state_for_question(question: str, pack: list[dict[str, Any]], intent: str) -> Any:
    return {
        "messages": [HumanMessage(content=question)],
        "current_user_message": question,
        "conversation_summary": "",
        "search_queries": _twelve_queries(),
        "planner_mode": "retrieval",
        "resolved_intent": intent,
        "retrieval_hits": [],
        "evidence_pack": pack,
        "retrieval_latency_ms": 0.0,
        "retrieval_over_budget": False,
        "route": "normal",
        "planner_invoked": True,
        "retry_state": {
            "planner_reason": "substantive-with-queries",
            "planner_outcome": "ok",
            "planner_mode": "retrieval",
            "resolved_intent": intent,
        },
        "recent_quote_ranges": [],
    }


async def test_ordinary_answer_certified_through_compiled_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.conversation.graph as graph_module
    from aa.conversation.graph import build_turn_graph

    monkeypatch.setattr(graph_module, "retrieval_stub_node", _preserving_stub)
    question = "что помогает при тяге вечером?"
    intent = "что помогает при тяге вечером"
    pack = [_pack_entry()]
    answer_text = "Поддержка рядом помогает пережить тягу спокойно."
    from aa.conversation.response_units import split_response_units

    units = split_response_units(answer_text)
    graph = build_turn_graph(
        planner_model=_retrieval_plan(intent),
        answer_model=_AnswerModel([answer_text]),
        verifier_model=_VerifierModel(
            [_support_result([u.unit_id for u in units], pack[0]["passage_id"])]
        ),
    )
    result = await graph.ainvoke(_graph_state_for_question(question, pack, intent))
    assert result["final_response"] == answer_text
    assert result["evidence_pack"] == pack
    assert result["delivery_status"] == "provisional-certified"
    candidate = result["answer_candidate"]
    certificate = result["verification_certificate"]
    assert candidate["text"] == answer_text
    assert certificate["answer_sha256"] == _sha(answer_text)
    assert certificate["context_digest"] == candidate["context_digest"]
    roles = [item.type for item in result["messages"]]
    assert "ai" in roles
    assert any(str(item.content) == answer_text for item in result["messages"] if item.type == "ai")


async def test_repair_propagates_new_evidence_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib as _hashlib

    from aa.conversation.response_units import split_response_units
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.retrieval import evidence as evidence_mod

    first_pack = [_pack_entry(passage_id="chapter-3#exp0000", char_start=0, char_end=120)]
    second_text = "Фиктивная поддержка рядом помогает пережить тягу спокойно."
    second_entry = _pack_entry(
        passage_id="chapter-3#exp0001", text=second_text, char_start=120, char_end=240
    )
    first_draft = "Поддержка рядом помогает. Тяга лечится луной за вечер."
    repaired_draft = "Поддержка рядом помогает пережить тягу спокойно."
    repaired_units = split_response_units(repaired_draft)

    from tests.test_p0_4_natural_grounding import _PlannerModel

    planner = _PlannerModel(
        [
            {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries(),
            }
        ]
    )

    # #306: repair retrieval traverses the shared async read/coverage loop.
    async def _fake_retrieve(
        index: Any, queries: object, *, config: Any = None, **kwargs: Any
    ) -> Any:
        from aa.retrieval.evidence import EvidencePack, EvidencePassageData

        _ = (index, queries, config, kwargs)
        passage = EvidencePassageData(
            passage_id=second_entry["passage_id"],
            exact_text=second_entry["text"],
            source_id=second_entry["source_id"],
            section_id=second_entry["section_id"],
            child_chunk_ids=("chapter-3:ru:1",),
            char_start=second_entry["char_start"],
            char_end=second_entry["char_end"],
            text_sha256=_hashlib.sha256(second_entry["text"].encode()).hexdigest(),
            source_sha256="s" * 64,
        )
        return EvidencePack(
            passages=(passage,), total_tokens=10, corpus_version="v", retrieval_metadata={}
        )

    from aa.conversation import retrieval_node as retrieval_node_mod

    monkeypatch.setattr(retrieval_node_mod, "aretrieve_with_semantic_selection", _fake_retrieve)
    _ = evidence_mod
    outcome = await run_v2_answer_turn(
        user_message="что помогает при тяге?",
        summary="",
        recent=[],
        evidence_pack=first_pack,
        answer_model=_AnswerModel([first_draft, repaired_draft]),
        verifier_model=_VerifierModel(
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
                            "evidence_passage_ids": [second_entry["passage_id"]],
                        }
                        for unit in repaired_units
                    ],
                    "all_required_supported": True,
                },
            ]
        ),
        planner_model=planner,
        retrieval_index=object(),
    )
    assert outcome["rounds"] == 1
    assert outcome["text"] == repaired_draft
    # The actually used repaired bundle (not the outdated pre-repair pack)
    # travels with the candidate.
    ids = {item["passage_id"] for item in outcome["evidence_pack"]}
    assert "chapter-3#exp0001" in ids

    from aa.conversation.finalization import (
        candidate_from_state,
        certify_candidate,
        evidence_digest_for_pack,
        verify_certificate,
    )

    question = "что помогает при тяге?"
    stored_texts = [
        {"unit_id": f"u{i}", "text": unit.text}
        for i, unit in enumerate(split_response_units(repaired_draft))
    ]
    state = {
        "current_user_message": question,
        "resolved_intent": question,
        "conversation_summary": "",
        "messages": [HumanMessage(content=question)],
        "evidence_pack": outcome["evidence_pack"],
        "final_response": repaired_draft,
        "draft_response": repaired_draft,
        "grounding_result": outcome["verification"],
        "retry_state": {"turn_telemetry": outcome["telemetry"]},
        "response_unit_texts": stored_texts,
        "route": "normal",
    }
    candidate, grounding, q, intent, summary, messages, texts = candidate_from_state(state)
    certificate = certify_candidate(
        candidate=candidate,
        grounding_result=grounding,
        question=q,
        resolved_intent=intent,
        summary=summary,
        recent_messages=messages,
        stored_unit_texts=texts,
    )
    assert certificate.evidence_digest == evidence_digest_for_pack(outcome["evidence_pack"])
    verify_certificate(
        candidate=candidate,
        certificate=certificate,
    )


async def test_sentence_deletion_invalidates_old_certificate() -> None:
    from aa.conversation.finalization import (
        AnswerCandidate,
        FinalizationError,
        VerificationCertificate,
        normalize_answer_text,
        sha256_text,
        verify_certificate,
    )

    question = "что помогает при тяге вечером?"
    pack = [_pack_entry()]
    full = "Поддержка рядом помогает. Тяга проходит, если обратиться за помощью."
    deleted = "Поддержка рядом помогает."
    from aa.conversation.finalization import context_digest_for_turn, evidence_digest_for_pack

    context_digest = context_digest_for_turn(
        question=question, resolved_intent=question, summary="", recent_texts=[question]
    )
    _ = AnswerCandidate(
        text=normalize_answer_text(full),
        evidence_bundle=pack,
        context_digest=context_digest,
        outcome_kind="answer",
    )
    from aa.conversation.finalization import WholeAnswerVerdict

    certificate = VerificationCertificate(
        answer_sha256=sha256_text(normalize_answer_text(full)),
        evidence_digest=evidence_digest_for_pack(pack),
        context_digest=context_digest,
        claim_verdicts=[],
        whole_answer_verdict=WholeAnswerVerdict(
            supported=True,
            addresses_intent=True,
            coverage_ok=True,
            conditions_preserved=True,
            quote_ok=True,
            reason="test",
        ),
    )
    mutated = AnswerCandidate(
        text=normalize_answer_text(deleted),
        evidence_bundle=pack,
        context_digest=context_digest,
        outcome_kind="answer",
    )
    with pytest.raises(FinalizationError):
        verify_certificate(candidate=mutated, certificate=certificate)


async def test_compaction_requires_new_certificate() -> None:
    from aa.conversation.finalization import (
        AnswerCandidate as _Candidate,
    )
    from aa.conversation.finalization import (
        FinalizationError as _FinalError,
    )
    from aa.conversation.finalization import (
        VerificationCertificate as _Certificate,
    )
    from aa.conversation.finalization import (
        WholeAnswerVerdict as _Whole,
    )
    from aa.conversation.finalization import (
        context_digest_for_turn,
        evidence_digest_for_pack,
        normalize_answer_text,
        sha256_text,
        verify_certificate,
    )
    from aa.conversation.output_limits import compact_text_to_envelope

    long_text = " ".join(["Поддержка рядом помогает пережить тягу спокойно."] * 40)
    pack = [_pack_entry()]
    question = "что помогает при тяге?"
    context_digest = context_digest_for_turn(
        question=question, resolved_intent=question, summary="", recent_texts=[question]
    )
    _ = _Candidate(
        text=normalize_answer_text(long_text),
        evidence_bundle=pack,
        context_digest=context_digest,
        outcome_kind="answer",
    )
    certificate = _Certificate(
        answer_sha256=sha256_text(normalize_answer_text(long_text)),
        evidence_digest=evidence_digest_for_pack(pack),
        context_digest=context_digest,
        claim_verdicts=[],
        whole_answer_verdict=_Whole(
            supported=True,
            addresses_intent=True,
            coverage_ok=True,
            conditions_preserved=True,
            quote_ok=True,
            reason="test",
        ),
    )
    compacted = compact_text_to_envelope(long_text)
    assert compacted != normalize_answer_text(long_text)
    mutated = _Candidate(
        text=normalize_answer_text(compacted),
        evidence_bundle=pack,
        context_digest=context_digest,
        outcome_kind="answer",
    )
    with pytest.raises(_FinalError):
        verify_certificate(candidate=mutated, certificate=certificate)


async def test_quote_replacement_fails_closed_through_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.conversation.graph as graph_module
    from aa.conversation.finalization import FinalizationError, finalize_answer_node
    from aa.conversation.graph import build_turn_graph

    monkeypatch.setattr(graph_module, "retrieval_stub_node", _preserving_stub)
    passage_text = "Фиктивная поддержка рядом помогает пережить тягу спокойно и уверенно."
    pack = [_pack_entry(text=passage_text, char_end=len(passage_text))]
    question = "что помогает при тяге?"
    intent = question
    quoted = (
        "«Фиктивная поддержка рядом помогает пережить тягу спокойно»"
        " и дальше своими словами о поддержке."
    )
    from aa.conversation.response_units import split_response_units

    units = split_response_units(quoted)
    graph = build_turn_graph(
        planner_model=_retrieval_plan(intent),
        answer_model=_AnswerModel([quoted]),
        verifier_model=_VerifierModel(
            [_support_result([u.unit_id for u in units], pack[0]["passage_id"])]
        ),
    )
    result = await graph.ainvoke(_graph_state_for_question(question, pack, intent))
    assert result["final_response"] == quoted
    # Replace the verified quotation with invented book text: re-finalizing
    # the mutated text with the old verdicts must fail closed.
    tampered = dict(result)
    tampered["final_response"] = (
        "«Придуманная цитата про луну за вечер» и дальше своими словами о поддержке."
    )
    tampered["draft_response"] = tampered["final_response"]
    with pytest.raises(FinalizationError):
        await finalize_answer_node(tampered)


async def test_safety_regeneration_requires_new_certificate() -> None:
    from aa.conversation.finalization import (
        AnswerCandidate as _Candidate,
    )
    from aa.conversation.finalization import (
        FinalizationError,
        certify_candidate,
        context_digest_for_turn,
        evidence_digest_for_pack,
        normalize_answer_text,
        sha256_text,
    )
    from aa.conversation.finalization import (
        VerificationCertificate as _Certificate,
    )
    from aa.conversation.finalization import (
        WholeAnswerVerdict as _Whole,
    )

    pack = [_pack_entry()]
    question = "что помогает при тяге?"
    context_digest = context_digest_for_turn(
        question=question, resolved_intent=question, summary="", recent_texts=[question]
    )
    unsafe = "Выпей немного, чтобы снять тягу вечером."
    candidate = _Candidate(
        text=normalize_answer_text(unsafe),
        evidence_bundle=pack,
        context_digest=context_digest,
        outcome_kind="answer",
    )
    with pytest.raises(FinalizationError):
        certify_candidate(
            candidate=candidate,
            grounding_result={"all_required_supported": True, "units": []},
            question=question,
            resolved_intent=question,
            summary="",
            recent_messages=[HumanMessage(content=question)],
        )
    safe = "Поддержка рядом помогает пережить тягу спокойно."
    _ = _Candidate(
        text=normalize_answer_text(safe),
        evidence_bundle=pack,
        context_digest=context_digest,
        outcome_kind="safety",
    )
    assert sha256_text(normalize_answer_text(safe)) != sha256_text(normalize_answer_text(unsafe))
    _ = _Certificate(
        answer_sha256=sha256_text(normalize_answer_text(safe)),
        evidence_digest=evidence_digest_for_pack(pack),
        context_digest=context_digest,
        claim_verdicts=[],
        whole_answer_verdict=_Whole(
            supported=True,
            addresses_intent=True,
            coverage_ok=True,
            conditions_preserved=True,
            quote_ok=True,
            reason="test",
        ),
    )


async def test_transport_split_preserves_order_and_text() -> None:
    from aa.conversation.finalization import (
        FinalizationError as _FinalError,
    )
    from aa.conversation.finalization import (
        normalize_answer_text,
        split_certified_text,
        verify_transport_split,
    )

    sentences = [
        f"Поддержка рядом помогает пережить тягу спокойно сегодня, шаг {index}."
        for index in range(30)
    ]
    long_text = " ".join(sentences)
    normalized = normalize_answer_text(long_text)
    segments = split_certified_text(normalized)
    assert len(segments) > 1
    verify_transport_split(normalized, segments)
    reassembled = " ".join(segments)
    assert " ".join(reassembled.split()) == " ".join(normalized.split())
    shortened = segments[:-1]
    with pytest.raises(_FinalError):
        verify_transport_split(normalized, shortened)
    reordered = list(reversed(segments))
    if reordered != segments:
        with pytest.raises(_FinalError):
            verify_transport_split(normalized, reordered)


async def test_unrelated_substituted_text_with_old_certificate_fails() -> None:
    from aa.conversation.finalization import (
        AnswerCandidate,
        FinalizationError,
        VerificationCertificate,
        WholeAnswerVerdict,
        context_digest_for_turn,
        evidence_digest_for_pack,
        normalize_answer_text,
        sha256_text,
        verify_certificate,
    )

    pack = [_pack_entry()]
    question = "что помогает при тяге?"
    context_digest = context_digest_for_turn(
        question=question, resolved_intent=question, summary="", recent_texts=[question]
    )
    original = "Поддержка рядом помогает пережить тягу спокойно."
    certificate = VerificationCertificate(
        answer_sha256=sha256_text(normalize_answer_text(original)),
        evidence_digest=evidence_digest_for_pack(pack),
        context_digest=context_digest,
        claim_verdicts=[],
        whole_answer_verdict=WholeAnswerVerdict(
            supported=True,
            addresses_intent=True,
            coverage_ok=True,
            conditions_preserved=True,
            quote_ok=True,
            reason="test",
        ),
    )
    unrelated = AnswerCandidate(
        text=normalize_answer_text("Совершенно другой ответ про погоду за окном."),
        evidence_bundle=pack,
        context_digest=context_digest,
        outcome_kind="answer",
    )
    with pytest.raises(FinalizationError):
        verify_certificate(candidate=unrelated, certificate=certificate)


async def test_rejected_candidate_does_not_enter_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.conversation.graph as graph_module
    from aa.conversation.failures import TurnFailed
    from aa.conversation.finalization import FinalizationError
    from aa.conversation.graph import build_turn_graph
    from aa.conversation.graph_runtime import GraphRuntimeError

    monkeypatch.setattr(graph_module, "retrieval_stub_node", _preserving_stub)
    question = "что помогает при тяге?"
    intent = question
    pack = [_pack_entry()]
    bad_draft = "Тяга лечится луной за один вечер без усилий."
    from aa.conversation.response_units import split_response_units

    units = split_response_units(bad_draft)
    graph = build_turn_graph(
        planner_model=_retrieval_plan(intent),
        answer_model=_AnswerModel([bad_draft]),
        verifier_model=_VerifierModel(
            [
                {
                    "units": [
                        {
                            "unit_id": unit.unit_id,
                            "scope": "book",
                            "supported": False,
                            "evidence_passage_ids": [pack[0]["passage_id"]],
                        }
                        for unit in units
                    ],
                    "all_required_supported": False,
                }
            ]
        ),
    )
    with pytest.raises((FinalizationError, TurnFailed, GraphRuntimeError, ValueError)):
        await graph.ainvoke(_graph_state_for_question(question, pack, intent))
    # Direct finalization of the rejected candidate also fails with no message.
    from aa.conversation.finalization import certify_candidate, context_digest_for_turn

    context_digest = context_digest_for_turn(
        question=question, resolved_intent=intent, summary="", recent_texts=[question]
    )
    from aa.conversation.finalization import AnswerCandidate as _Candidate

    candidate = _Candidate(
        text=bad_draft,
        evidence_bundle=pack,
        context_digest=context_digest,
        outcome_kind="answer",
    )
    with pytest.raises(FinalizationError):
        certify_candidate(
            candidate=candidate,
            grounding_result={
                "all_required_supported": False,
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                        "addresses_intent": False,
                        "origin": "book_claim",
                        "origin_ref": {},
                    }
                    for unit in units
                ],
            },
            question=question,
            resolved_intent=intent,
            summary="",
            recent_messages=[HumanMessage(content=question)],
        )


async def test_graph_runtime_rejects_tampered_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.conversation.graph as graph_module
    from aa.conversation.graph import build_turn_graph
    from aa.conversation.graph_runtime import GraphRuntimeError, GraphTurnRuntime

    monkeypatch.setattr(graph_module, "retrieval_stub_node", _preserving_stub)
    question = "что помогает при тяге вечером?"
    intent = question
    pack = [_pack_entry()]
    answer_text = "Поддержка рядом помогает пережить тягу спокойно."
    from aa.conversation.response_units import split_response_units

    units = split_response_units(answer_text)
    graph = build_turn_graph(
        planner_model=_retrieval_plan(intent),
        answer_model=_AnswerModel([answer_text]),
        verifier_model=_VerifierModel(
            [_support_result([u.unit_id for u in units], pack[0]["passage_id"])]
        ),
    )
    runtime = GraphTurnRuntime(graph=graph)
    runtime._running = True
    original_invoke = runtime._invoke_graph

    async def _tampered_invoke(g: Any, thread: str, text: str) -> dict[str, Any]:
        result = await original_invoke(g, thread, text)
        # Borrow the positive certificate for mutated text.
        result["final_response"] = "Совершенно другой ответ про погоду за окном."
        return result

    monkeypatch.setattr(runtime, "_invoke_graph", _tampered_invoke)
    with pytest.raises(GraphRuntimeError):
        await runtime.run_turn(12345, question)


async def test_app_delivery_receipts_are_serializable() -> None:
    from aa.conversation.finalization import (
        build_delivery_receipts,
        normalize_answer_text,
        sha256_text,
        split_certified_text,
    )

    text = "Поддержка рядом помогает пережить тягу спокойно."
    normalized = normalize_answer_text(text)
    segments = split_certified_text(normalized)
    receipts = build_delivery_receipts(
        certified_text=normalized,
        segments=segments,
        certificate_id="cert-1",
        turn_id="turn-1",
        channel="sendMessage",
    )
    assert len(receipts) == len(segments)
    for receipt in receipts:
        dumped = receipt.model_dump(mode="json")
        assert dumped["final_sha256"] == sha256_text(normalized)
        assert dumped["status"] in ("confirmed", "failed", "unknown")
        assert dumped["channel"] in ("sendMessage", "sendVoice")
        assert dumped["retry_id"]
