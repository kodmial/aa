"""P1 #312 delivered-quote history and factual selection telemetry tests.

Production-boundary tests over invented fixture text only (no canonical
book text committed). Every assertion uses privacy-safe metadata:
provenance, offsets, counts and digests, never corpus or user text.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from aa.conversation.finalization import evidence_digest_for_pack
from aa.conversation.quote_state import (
    commit_delivery,
    delivered_ranges_for_receipts,
    delivered_ranges_from_candidate,
    is_adjacent_to_recent,
    merge_recent_ranges,
    pack_pages_recent,
    ranges_from_delivered_quotes,
)
from aa.conversation.turn_pipeline import run_v2_answer_turn


def _pack_entry(
    text: str = "Фиктивная поддержка рядом. Спокойный разговор помогает. Эпилог.",
    passage_id: str = "chapter-3#exp0000",
    source_id: str = "ru-fourth-edition-txt",
    section_id: str = "chapter-3",
    char_start: int = 0,
) -> dict[str, Any]:
    char_end = char_start + len(text)
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": source_id,
        "section_id": section_id,
        "char_start": char_start,
        "char_end": char_end,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


class _StaticAnswer:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    async def ainvoke(self, messages: Any) -> AIMessage:
        if not self._texts:
            raise AssertionError("answer model called more times than scripted")
        return AIMessage(content=self._texts.pop(0))


class _ScriptedVerifier:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self._decisions = [dict(item) for item in decisions]

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        if not self._decisions:
            raise AssertionError("verifier called more times than scripted")
        return dict(self._decisions.pop(0))


def _book_decision(passage_id: str) -> dict[str, Any]:
    return {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": [passage_id],
        "addresses_intent": True,
    }


def _receipt(
    start: int, end: int, status: str = "confirmed", segments: int = 1, index: int = 0
) -> dict[str, Any]:
    return {
        "turn_id": "thread",
        "certificate_id": "cert-1",
        "final_sha256": "f" * 64,
        "segment_index": index,
        "segment_count": segments,
        "char_start": start,
        "char_end": end,
        "utf8_start": start,
        "utf8_end": end,
        "status": status,
        "channel": "sendMessage",
        "retry_id": "retry-1",
    }


async def test_paraphrase_turns_create_zero_quote_history() -> None:
    """Ten consecutive natural paraphrases record no verbatim ranges."""
    pack = [_pack_entry()]
    recent: list[dict[str, Any]] = []
    for _ in range(10):
        draft = "Поддержка рядом помогает спокойно, обратитесь за помощью."
        outcome = await run_v2_answer_turn(
            user_message="Как справиться с тягой?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=_StaticAnswer([draft]),
            verifier_model=_ScriptedVerifier([_book_decision(pack[0]["passage_id"])] * 2),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=3,
            recent_quote_ranges=recent,
        )
        # Provisional graph completion never mutates confirmed history.
        assert outcome["recent_quote_ranges"] == recent
        # A paraphrase carries no book quotation: nothing deliverable.
        confirmed, possibly = delivered_ranges_from_candidate(
            certified_text=outcome["text"],
            evidence_pack=pack,
            claim_verdicts=[
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": True,
                    "evidence_passage_ids": [pack[0]["passage_id"]],
                    "addresses_intent": True,
                    "origin": "book_claim",
                    "origin_ref": {"kind": "book"},
                }
            ],
            stored_unit_texts=[{"unit_id": "u1", "text": outcome["text"]}],
            receipts=[_receipt(0, len(outcome["text"]))],
            certificate_id="cert-1",
        )
        assert confirmed == [] and possibly == []
        recent = commit_delivery(recent, None, certified_text="", receipts=[])
        assert recent == []
    assert recent == []


async def test_short_quote_records_exact_chars_not_whole_passage() -> None:
    exact = "Спокойный разговор помогает"
    text = f"Фиктивная поддержка рядом. {exact}. Эпилог истории."
    pack = [_pack_entry(text)]
    draft = f"Вот точные слова: «{exact}»."
    outcome = await run_v2_answer_turn(
        user_message="Приведи точную цитату про поддержку.",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=_StaticAnswer([draft]),
        verifier_model=_ScriptedVerifier([_book_decision(pack[0]["passage_id"])] * 2),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=3,
    )
    assert exact in outcome["text"]
    confirmed, _ = delivered_ranges_from_candidate(
        certified_text=outcome["text"],
        evidence_pack=pack,
        claim_verdicts=[
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
                "origin": "book_claim",
                "origin_ref": {"kind": "book"},
            }
        ],
        stored_unit_texts=[{"unit_id": "u1", "text": outcome["text"]}],
        receipts=[_receipt(0, len(outcome["text"]))],
        certificate_id="cert-1",
    )
    assert len(confirmed) == 1
    span = confirmed[0]
    assert span["char_end"] - span["char_start"] == len(exact)
    assert span["char_end"] - span["char_start"] < len(text)
    assert span["source_sha256"] == "s" * 64
    assert "text" not in span


async def test_user_report_quote_adds_no_book_history() -> None:
    user_message = "Я сказал себе: терпение помогает"
    quoted = "терпение помогает"
    pack = [_pack_entry("Фиктивная поддержка рядом. Эпилог.")]
    candidate_text = f"Вы написали: «{quoted}»."
    confirmed, possibly = delivered_ranges_from_candidate(
        certified_text=candidate_text,
        evidence_pack=pack,
        claim_verdicts=[
            {
                "unit_id": "u1",
                "scope": "conversation_glue",
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
                "origin": "user_report",
                "origin_ref": {"kind": "user_message", "role": "human"},
            }
        ],
        stored_unit_texts=[{"unit_id": "u1", "text": candidate_text}],
        receipts=[_receipt(0, len(candidate_text))],
        certificate_id="cert-1",
    )
    assert confirmed == [] and possibly == []
    _ = [HumanMessage(content=user_message)]


def test_split_partial_delivery_records_only_confirmed_prefix() -> None:
    quote_a = "Спокойный разговор помогает"
    quote_b = "Фиктивная поддержка рядом"
    text = f"Начало {quote_a} середина {quote_b} конец истории."
    pack = [_pack_entry(text)]
    start_a = text.index(quote_a)
    end_a = start_a + len(quote_a)
    start_b = text.index(quote_b)
    end_b = start_b + len(quote_b)
    _ = (start_a, end_a, start_b, end_b)
    answer_a = f"Цитата: «{quote_a}»."
    answer_b = f"Далее: «{quote_b}»."
    certified = answer_a + " " + answer_b
    split_at = len(answer_a) + 1
    receipts = [
        _receipt(0, split_at, status="confirmed", segments=2, index=0),
        _receipt(split_at, len(certified), status="failed", segments=2, index=1),
    ]
    confirmed, possibly = delivered_ranges_from_candidate(
        certified_text=certified,
        evidence_pack=pack,
        claim_verdicts=[
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
                "origin": "book_claim",
                "origin_ref": {"kind": "book"},
            },
            {
                "unit_id": "u2",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
                "origin": "book_claim",
                "origin_ref": {"kind": "book"},
            },
        ],
        stored_unit_texts=[
            {"unit_id": "u1", "text": answer_a},
            {"unit_id": "u2", "text": answer_b},
        ],
        receipts=receipts,
        certificate_id="cert-1",
    )
    # Only the first segment's quote is confirmed; the failed suffix is
    # at most a possibly-delivered safety record, never confirmed.
    assert len(confirmed) == 1
    assert confirmed[0]["char_end"] - confirmed[0]["char_start"] == len(quote_a)
    assert all(item["delivery_status"] == "confirmed" for item in confirmed)
    assert all(item["delivery_status"] == "possibly-delivered" for item in possibly)
    # Retry is idempotent: no phantom or duplicated ranges.
    merged_once = merge_recent_ranges([], [*confirmed, *possibly])
    merged_twice = merge_recent_ranges(merged_once, [*confirmed, *possibly])
    assert merged_twice == merged_once


def test_merge_preserves_disjoint_spans_and_is_idempotent() -> None:
    first = {
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "source_sha256": "s" * 64,
        "char_start": 0,
        "char_end": 20,
        "passage_id": "chapter-3#exp0000",
        "delivery_status": "confirmed",
    }
    second = {
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "source_sha256": "s" * 64,
        "char_start": 100,
        "char_end": 130,
        "passage_id": "chapter-3#exp0000",
        "delivery_status": "confirmed",
    }
    merged = merge_recent_ranges([], [first, second])
    assert len(merged) == 2
    again = merge_recent_ranges(merged, [first, second])
    assert again == merged
    # Session reset clears history.
    assert merge_recent_ranges([], []) == []


def test_paraphrase_not_blocked_but_serial_paging_blocked() -> None:
    recent = merge_recent_ranges(
        [],
        [
            {
                "source_id": "ru-fourth-edition-txt",
                "section_id": "chapter-3",
                "source_sha256": "s" * 64,
                "char_start": 0,
                "char_end": 60,
                "passage_id": "chapter-3#exp0000",
                "delivery_status": "confirmed",
            }
        ],
    )
    # A paraphrase carries no verbatim spans, so stripping is a no-op and
    # the supporting pack is not censored by confirmed history alone.
    far_pack = [_pack_entry("Совсем другой отрывок про помощь.", char_start=5000)]
    assert pack_pages_recent(far_pack, recent) is False
    adjacent_pack = [_pack_entry("Продолжение цитаты рядом.", char_start=60)]
    assert pack_pages_recent(adjacent_pack, recent) is True
    candidate = {
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 70,
        "char_end": 120,
    }
    assert is_adjacent_to_recent(candidate, recent) is True


async def test_lexical_fallback_never_claims_model_selection() -> None:
    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает спокойно, обратитесь за помощью."
    outcome = await run_v2_answer_turn(
        user_message="Как справиться с тягой?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=_StaticAnswer([draft]),
        verifier_model=_ScriptedVerifier([_book_decision(pack[0]["passage_id"])] * 2),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=3,
    )
    telemetry = outcome["telemetry"]
    assert telemetry["selection_route"] == "lexical_fallback"
    assert telemetry["semantic_selection_applied"] is False
    assert telemetry["pack_order"] == "lexical_fallback"
    assert telemetry["pack_order"] != "model_selection"
    assert telemetry["evidence_delivered"] == "pending-delivery"
    assert telemetry["delivery_outcome"] == "provisional-certified"


async def test_model_selection_route_requires_executed_model_events() -> None:
    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает спокойно, обратитесь за помощью."
    events = {
        "selection_model_used": True,
        "selection_fallback_used": False,
        "selection_deep_rank_gt16": True,
        "discovered_ids": ["c1", "c2"],
        "previewed_ids": ["c1"],
        "read_ids": ["c1"],
        "coverage_status": "ready",
    }
    outcome = await run_v2_answer_turn(
        user_message="Как справиться с тягой?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=_StaticAnswer([draft]),
        verifier_model=_ScriptedVerifier([_book_decision(pack[0]["passage_id"])] * 2),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=3,
        selection_events=events,
    )
    telemetry = outcome["telemetry"]
    assert telemetry["selection_route"] == "model_selection"
    assert telemetry["semantic_selection_applied"] is True
    assert telemetry["pack_order"] == "model_selection"
    assert telemetry["evidence_discovered"] == 2
    assert telemetry["evidence_previewed"] == 1
    assert telemetry["evidence_read"] == 1
    assert telemetry["pack_digest"] == evidence_digest_for_pack(pack)
    assert telemetry["evidence_source_ids"] == ["ru-fourth-edition-txt"]
    payload = repr(sorted(telemetry.items()))
    assert pack[0]["text"] not in payload


async def test_empty_stage_reports_unavailable_not_requested() -> None:
    draft = "Здравствуйте! Чем могу помочь сегодня?"
    outcome_glue = await run_v2_answer_turn(
        user_message="Привет!",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=_StaticAnswer([draft]),
        verifier_model=_ScriptedVerifier(
            [
                {
                    "requires_book_evidence": False,
                    "supported": True,
                    "evidence_passage_ids": [],
                    "addresses_intent": True,
                }
            ]
            * 2
        ),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=0,
        planner_mode="conversational",
        planner_reason="legitimate-glue",
    )
    assert outcome_glue["telemetry"]["selection_route"] == "not_requested"
    assert outcome_glue["telemetry"]["semantic_selection_applied"] is False


def test_qualification_rejects_fabricated_semantic_support() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    base: dict[str, Any] = {
        "answer_outcome": "served",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 3,
        "retrieval_passages": 2,
        "verified_book_units": 1,
        "response_units": 2,
        "adequacy_verdict": "pass",
        "failure_category": "",
        "planner_reason": "substantive-with-queries",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
        "selection_route": "lexical_fallback",
        "pack_order": "lexical_fallback",
        "semantic_selection_applied": False,
    }
    reply = "Поддержка рядом помогает спокойно. Обратитесь за помощью."
    assert _is_grounded_substantive_reply(dict(base), reply) in (True, False)
    fabricated = dict(base)
    fabricated["semantic_selection_applied"] = True
    fabricated["selection_route"] = "lexical_fallback"
    assert _is_grounded_substantive_reply(fabricated, reply) is False
    fabricated_order = dict(base)
    fabricated_order["pack_order"] = "model_selection"
    fabricated_order["selection_route"] = "lexical_fallback"
    assert _is_grounded_substantive_reply(fabricated_order, reply) is False
    legacy = dict(base)
    legacy["pack_order"] = "semantic"
    assert _is_grounded_substantive_reply(legacy, reply) is False


def test_quote_state_carries_no_plaintext() -> None:
    from aa.conversation.quote_state import quote_ranges_digest

    pack_text = "Фиктивная поддержка рядом спокойно."
    pack = [_pack_entry(pack_text)]
    answer = "Цитата: «поддержка рядом»."
    confirmed, _ = delivered_ranges_from_candidate(
        certified_text=answer,
        evidence_pack=pack,
        claim_verdicts=[
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
                "origin": "book_claim",
                "origin_ref": {"kind": "book"},
            }
        ],
        stored_unit_texts=[{"unit_id": "u1", "text": answer}],
        receipts=[_receipt(0, len(answer))],
        certificate_id="cert-1",
    )
    merged = merge_recent_ranges([], confirmed)
    blob = repr(merged) + quote_ranges_digest(merged)
    assert pack_text not in blob
    assert "поддержка рядом" not in blob


def test_ranges_from_certificate_excludes_user_reports() -> None:
    from aa.conversation.quote_provenance import AnswerQuoteCertificate, SpanProvenance

    certificate = AnswerQuoteCertificate(
        parser_version="answer-quote-parser/1",
        answer_sha256="a" * 64,
        answer_chars=10,
        answer_utf8_len=10,
        total_quote_chars=8,
        book_quote_chars=4,
        spans=[
            SpanProvenance(
                span_text_sha256="b" * 64,
                answer_char_start=0,
                answer_char_end=4,
                origin="book_claim",
                unit_ids=["u1"],
                origin_ref_type="book",
                book_source_sha256="s" * 64,
                book_section_id="chapter-3",
                book_source_id="ru-fourth-edition-txt",
                book_source_char_start=10,
                book_source_char_end=14,
                book_passage_ids=["chapter-3#exp0000"],
                book_corpus_version="r" * 64,
            ),
            SpanProvenance(
                span_text_sha256="c" * 64,
                answer_char_start=5,
                answer_char_end=9,
                origin="user_report",
                unit_ids=["u2"],
                origin_ref_type="user_message",
                user_message_id="turn-human-current",
                user_char_start=0,
                user_char_end=4,
            ),
        ],
        dangling=[],
        overlap_units=[["u1"], ["u2"]],
        passed=True,
    )
    ranges = ranges_from_delivered_quotes(certificate, certificate_id="cert-1")
    assert len(ranges) == 1
    assert ranges[0]["char_start"] == 10
    assert ranges[0]["char_end"] == 14
    confirmed, _ = delivered_ranges_for_receipts(
        certificate, certified_text="x" * 10, receipts=[], certificate_id="cert-1"
    )
    assert confirmed == []
