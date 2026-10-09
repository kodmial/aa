"""AUDIT P0 (kodmial/aa#308): whole-answer exact quotes and claim-origin attribution.

Regression controls for two reproducible P0 failure modes:

- invented multi-sentence quoted book text bypassing exact-quote validation
  because quote detection happened after sentence splitting;
- faithfully quoted user words incorrectly required to be book citations.

All controls run deterministically (no model): the canonical whole-answer
quote parser plus the typed origin/provenance contract decide. Two
pipeline controls additionally exercise the compiled turn path.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aa.conversation.output_limits import QUOTE_BUDGET_CHARS, aggregate_quote_chars
from aa.conversation.quote_provenance import (
    QUOTE_PARSER_VERSION,
    anchor_book_span,
    anchor_user_span,
    book_quote_chars_for_answer,
    build_user_message_index,
    extract_answer_quotes,
    total_quoted_chars,
)
from aa.conversation.response_units import split_response_units
from aa.conversation.turn_pipeline import final_book_quote_chars as pipeline_book_quotes
from aa.conversation.verifier import (
    check_exact_quotes,
    check_quote_origins,
    coerce_single_verdict,
)
from aa.conversation.verifier_schema import (
    GroundingResult,
    VerifierValidationError,
    validate_grounding_result,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack_entry(
    text: str,
    *,
    passage_id: str = "chapter-3#range:0-64:deadbeef",
    source_id: str = "ru-fourth-edition-txt",
    section_id: str = "chapter-3",
    start: int = 0,
    source_sha: str = "s" * 64,
) -> dict[str, Any]:
    end = start + len(text)
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": source_id,
        "section_id": section_id,
        "child_chunk_ids": [passage_id],
        "char_start": start,
        "char_end": end,
        "text_sha256": _sha(text),
        "source_sha256": source_sha,
        "corpus_version": "r" * 64,
    }


def _assemble(
    units: Any,
    decisions: list[dict[str, Any]],
) -> GroundingResult:
    verdicts = [
        coerce_single_verdict(dict(decision), unit_id=unit.unit_id)
        for unit, decision in zip(units, decisions, strict=True)
    ]
    all_supported = all(verdict.supported for verdict in verdicts)
    return validate_grounding_result(
        {
            "verified": True,
            "units": [verdict.model_dump() for verdict in verdicts],
            "all_required_supported": all_supported,
        },
        expected_unit_ids=[unit.unit_id for unit in units],
    )


def _book_decision(passage_id: str, *, origin: str | None = "book_claim") -> dict[str, Any]:
    decision: dict[str, Any] = {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": [passage_id],
        "addresses_intent": True,
    }
    if origin is not None:
        decision["claim_origin"] = origin
    return decision


def _glue_decision() -> dict[str, Any]:
    return {
        "requires_book_evidence": False,
        "supported": True,
        "evidence_passage_ids": [],
        "addresses_intent": True,
    }


def _user_decision() -> dict[str, Any]:
    return {
        "requires_book_evidence": False,
        "supported": True,
        "evidence_passage_ids": [],
        "addresses_intent": True,
        "claim_origin": "user_report",
    }


def _certify(
    draft: str,
    decisions: list[dict[str, Any]],
    pack: list[dict[str, Any]],
    *,
    recent: list[Any] | None = None,
    user_message: str = "",
) -> Any:
    units = split_response_units(draft)
    assert len(units) == len(decisions), "decisions must cover every response unit"
    result = _assemble(units, decisions)
    check_exact_quotes(units=units, result=result, passages=pack, answer_text=draft)
    return check_quote_origins(
        answer=draft,
        units=units,
        result=result,
        passages=pack,
        recent=list(recent or []),
        user_message=user_message,
    )


# ---------------------------------------------------------------------------
# Whole-answer exact-quotation controls.
# ---------------------------------------------------------------------------


def test_invented_multisentence_quote_fails_despite_positive_verdict() -> None:
    """An invented second sentence fails even with a positive model verdict."""
    book = "Преамбула про поддержку. Первое предложение. Эпилог про трезвость."
    pack = [_pack_entry(book)]
    draft = "В книге написано: «Первое предложение. Второе предложение»."
    units = split_response_units(draft)
    assert len(units) == 2  # razdel splits inside the quotation (the old bypass)
    result = _assemble(units, [_book_decision(pack[0]["passage_id"])] * 2)
    with pytest.raises(VerifierValidationError):
        check_exact_quotes(units=units, result=result, passages=pack)
    with pytest.raises(VerifierValidationError):
        check_quote_origins(
            answer=draft, units=units, result=result, passages=pack, user_message="что там?"
        )


def test_exact_two_sentence_quote_passes_with_source_anchors() -> None:
    """A real canonical two-sentence quotation passes with exact anchors."""
    book = "Преамбула. Первое предложение. Второе предложение. Эпилог."
    book_nl = "Преамбула. Первое предложение.\nВторое предложение. Эпилог."
    for quoted, book_for_span in (
        ("В книге написано: «Первое предложение. Второе предложение».", book),
        ("В книге написано: «Первое предложение.\nВторое предложение».", book_nl),
    ):
        pack_variant = [_pack_entry(book_for_span)]
        certificate = _certify(
            quoted,
            [_book_decision(pack_variant[0]["passage_id"])] * 2,
            pack_variant,
            user_message="что написано?",
        )
        assert certificate.passed is True
        assert certificate.parser_version == QUOTE_PARSER_VERSION
        assert len(certificate.spans) == 1
        span = certificate.spans[0]
        assert span.origin == "book_claim"
        assert span.origin_ref_type == "book"
        assert span.book_passage_ids == [pack_variant[0]["passage_id"]]
        assert span.book_source_sha256 == "s" * 64
        assert span.book_section_id == "chapter-3"
        assert span.book_source_id == "ru-fourth-edition-txt"
        # Exact answer offsets round-trip; UTF-8 offsets are explicit.
        raw = quoted.encode("utf-8")
        assert quoted[span.answer_char_start : span.answer_char_end] in (
            "Первое предложение. Второе предложение",
            "Первое предложение.\nВторое предложение",
        )
        assert (
            raw[span.answer_utf8_start : span.answer_utf8_end].decode("utf-8")
            == quoted[span.answer_char_start : span.answer_char_end]
        )
        # Exact source range round-trips against the contiguous run.
        run_text = book_for_span
        start = span.book_source_char_start - pack_variant[0]["char_start"]
        end = span.book_source_char_end - pack_variant[0]["char_start"]
        assert run_text[start:end] == quoted[span.answer_char_start : span.answer_char_end]
        assert certificate.book_quote_chars == len(
            quoted[span.answer_char_start : span.answer_char_end]
        )


def test_one_word_substitution_fails() -> None:
    book = "Преамбула. Первое предложение. Второе предложение. Эпилог."
    pack = [_pack_entry(book)]
    draft = "В книге написано: «Первое предложение. Второе утверждение»."
    units = split_response_units(draft)
    result = _assemble(units, [_book_decision(pack[0]["passage_id"])] * len(units))
    with pytest.raises(VerifierValidationError):
        check_exact_quotes(units=units, result=result, passages=pack)


def test_changed_negation_fails() -> None:
    book = "Трезвость приходит постепенно через честный разбор."
    pack = [_pack_entry(book)]
    draft = "В книге написано: «Трезвость не приходит постепенно»."
    units = split_response_units(draft)
    result = _assemble(units, [_book_decision(pack[0]["passage_id"])] * len(units))
    with pytest.raises(VerifierValidationError):
        check_exact_quotes(units=units, result=result, passages=pack)


def test_unclosed_quote_cannot_silently_pass() -> None:
    book = "Преамбула. Первое предложение. Эпилог."
    pack = [_pack_entry(book)]
    draft = "В книге написано: «Первое предложение."
    units = split_response_units(draft)
    result = _assemble(units, [_book_decision(pack[0]["passage_id"])] * len(units))
    with pytest.raises(VerifierValidationError):
        check_exact_quotes(units=units, result=result, passages=pack)
    with pytest.raises(VerifierValidationError):
        check_quote_origins(
            answer=draft, units=units, result=result, passages=pack, user_message="что там?"
        )


def test_partially_matched_quote_cannot_silently_pass() -> None:
    """Only the first sentence matches; the fabricated tail must fail."""
    book = "Преамбула. Первое предложение. Эпилог."
    pack = [_pack_entry(book)]
    draft = "В книге написано: «Первое предложение. Выдуманное продолжение»."
    units = split_response_units(draft)
    result = _assemble(units, [_book_decision(pack[0]["passage_id"])] * len(units))
    with pytest.raises(VerifierValidationError):
        check_exact_quotes(units=units, result=result, passages=pack)


# ---------------------------------------------------------------------------
# Claim-origin attribution controls.
# ---------------------------------------------------------------------------


def test_user_attribution_passes_without_book_passage() -> None:
    user_text = "не могу уснуть вторую ночь"
    recent = [HumanMessage(content=user_text)]
    draft = f"Вы написали «{user_text}». Что вы имеете в виду?"
    certificate = _certify(
        draft,
        [_user_decision(), _glue_decision()],
        [_pack_entry("Совершенно другой текст про поддержку.")],
        recent=recent,
        user_message=f"{user_text}, что делать?",
    )
    assert certificate.passed is True
    assert len(certificate.spans) == 1
    span = certificate.spans[0]
    assert span.origin == "user_report"
    assert span.origin_ref_type == "user_message"
    assert span.user_message_id.startswith("turn-human-")
    assert span.book_passage_ids == []
    assert certificate.book_quote_chars == 0
    assert certificate.total_quote_chars == len(user_text)


def test_user_attribution_without_matching_human_message_fails() -> None:
    draft = "Вы написали «не могу уснуть вторую ночь». Что вы имеете в виду?"
    units = split_response_units(draft)
    result = _assemble(units, [_user_decision(), _glue_decision()])
    pack = [_pack_entry("Совершенно другой текст про поддержку.")]
    # No HumanMessage carries the quoted words: attribution fails closed.
    with pytest.raises(VerifierValidationError):
        check_quote_origins(
            answer=draft, units=units, result=result, passages=pack, user_message="как дела?"
        )


def test_user_report_requiring_book_evidence_fails_closed() -> None:
    """A causal-inference verdict cannot travel as a harmless user report."""
    with pytest.raises(VerifierValidationError):
        coerce_single_verdict(
            {
                "requires_book_evidence": True,
                "supported": False,
                "evidence_passage_ids": [],
                "addresses_intent": False,
                "claim_origin": "user_report",
            },
            unit_id="u1",
        )


def test_advice_without_book_support_fails_closed() -> None:
    """Reworded advice with no cited passage fails closed (decision boundary)."""
    units = split_response_units("Вам следует полностью изменить жизнь сегодня.")
    # kodmial/aa#310: a supported book claim with empty citations fails at
    # the strict decision boundary, before the cite gate ever runs.
    with pytest.raises(VerifierValidationError):
        _assemble(
            units,
            [
                {
                    "requires_book_evidence": True,
                    "supported": True,
                    "evidence_passage_ids": [],
                    "addresses_intent": True,
                }
            ],
        )
    with pytest.raises(VerifierValidationError):
        coerce_single_verdict(
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            },
            unit_id="u1",
        )


def test_mixed_user_report_and_book_claim_cannot_launder() -> None:
    """One unit mixing user repetition with a new book quote must fail as user_report."""
    human = "не могу уснуть"
    book = "Трезвость приходит постепенно через честный разбор."
    pack = [_pack_entry(book)]
    draft = f"Вы написали «{human}», и «Трезвость приходит постепенно» помогает."
    units = split_response_units(draft)
    assert len(units) == 1
    result = _assemble(units, [_user_decision()])
    with pytest.raises(VerifierValidationError):
        check_quote_origins(
            answer=draft,
            units=units,
            result=result,
            passages=pack,
            recent=[HumanMessage(content=human)],
            user_message=human,
        )


def test_book_quote_missourced_as_user_message_fails() -> None:
    book = "Преамбула. Первое предложение. Эпилог."
    pack = [_pack_entry(book)]
    draft = "В книге написано: «Первое предложение»."
    units = split_response_units(draft)
    decision = _book_decision(pack[0]["passage_id"])
    decision["origin_ref"] = {"kind": "user_message", "message_id": "turn-human-current"}
    result = _assemble(units, [decision])
    with pytest.raises(VerifierValidationError):
        check_quote_origins(
            answer=draft,
            units=units,
            result=result,
            passages=pack,
            recent=[HumanMessage(content="Первое предложение")],
            user_message="Первое предложение",
        )


def test_book_quote_missourced_as_capability_or_policy_fails() -> None:
    book = "Преамбула. Первое предложение. Эпилог."
    pack = [_pack_entry(book)]
    draft = "В книге написано: «Первое предложение»."
    for kind in ("system", "capability", "policy"):
        units = split_response_units(draft)
        decision = _book_decision(pack[0]["passage_id"])
        decision["origin_ref"] = {"kind": kind}
        result = _assemble(units, [decision])
        with pytest.raises(VerifierValidationError):
            check_quote_origins(
                answer=draft, units=units, result=result, passages=pack, user_message="что там?"
            )


def test_user_report_citing_assistant_role_fails() -> None:
    with pytest.raises(VerifierValidationError):
        coerce_single_verdict(
            {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
                "claim_origin": "user_report",
                "origin_ref": {"kind": "user_message", "role": "assistant"},
            },
            unit_id="u1",
        )


def test_assistant_message_is_never_user_evidence() -> None:
    """An AIMessage carrying the same words is not a trusted user referent."""
    index = build_user_message_index(
        [AIMessage(content="до свидания"), HumanMessage(content="здравствуйте")],
        "здравствуйте",
    )
    assert all(entry["role"] == "human" for entry in index)
    assert anchor_user_span("до свидания", index) is None
    draft = "Вы написали «до свидания». Пока!"
    units = split_response_units(draft)
    result = _assemble(units, [_user_decision(), _glue_decision()])
    with pytest.raises(VerifierValidationError):
        check_quote_origins(
            answer=draft,
            units=units,
            result=result,
            passages=[_pack_entry("Другой текст.")],
            recent=[AIMessage(content="до свидания")],
            user_message="пока",
        )


def test_user_explanation_is_not_a_fresh_diagnosis() -> None:
    """A quoted user-reported explanation passes as user_report, not AA diagnosis."""
    user_text = "не могу уснуть из-за шума на работе"
    draft = f"Вы объяснили «{user_text}». Расскажите подробнее?"
    certificate = _certify(
        draft,
        [_user_decision(), _glue_decision()],
        [_pack_entry("Другой текст про поддержку.")],
        recent=[HumanMessage(content=user_text)],
        user_message=f"{user_text}, что делать?",
    )
    assert certificate.passed is True
    assert certificate.spans[0].origin == "user_report"
    assert certificate.book_quote_chars == 0


# ---------------------------------------------------------------------------
# Contiguous source ranges, parser contract, offsets, budgets.
# ---------------------------------------------------------------------------


def test_cross_chunk_contiguous_quote_passes() -> None:
    first = "Первое предложение. "
    second = "Второе предложение."
    pack = [
        _pack_entry(first, passage_id="chapter-3#c0001", start=0),
        _pack_entry(second, passage_id="chapter-3#c0002", start=len(first)),
    ]
    anchor = anchor_book_span("Первое предложение. Второе предложение", pack)
    assert anchor is not None
    assert anchor.passage_ids == ("chapter-3#c0001", "chapter-3#c0002")
    assert anchor.source_char_start == 0
    assert anchor.source_char_end == len(first + second) - len(".")
    draft = "В книге написано: «Первое предложение. Второе предложение»."
    certificate = _certify(
        draft,
        [_book_decision("chapter-3#c0001"), _book_decision("chapter-3#c0002")],
        pack,
        user_message="что там?",
    )
    assert certificate.passed is True


def test_noncontiguous_slices_are_not_silently_concatenated() -> None:
    first = "Первое предложение. "
    second = "Второе предложение."
    pack = [
        _pack_entry(first, passage_id="chapter-3#c0001", start=0),
        # Gap in canonical offsets: not one contiguous source range.
        _pack_entry(second, passage_id="chapter-3#c0009", start=500),
    ]
    assert anchor_book_span("Первое предложение. Второе предложение", pack) is None
    draft = "В книге написано: «Первое предложение. Второе предложение»."
    units = split_response_units(draft)
    result = _assemble(
        units,
        [_book_decision("chapter-3#c0001"), _book_decision("chapter-3#c0009")],
    )
    with pytest.raises(VerifierValidationError):
        check_exact_quotes(units=units, result=result, passages=pack)


def test_multiline_invented_quote_fails_exactness_and_budget() -> None:
    invented = "x" * 150 + "\n" + "y" * 151
    draft = f"В книге написано: «{invented}»."
    # The cross-line span counts toward the budget (no newline bypass).
    assert aggregate_quote_chars(draft) == len(invented) > QUOTE_BUDGET_CHARS
    pack = [_pack_entry("Настоящий текст про поддержку.")]
    units = split_response_units(draft)
    result = _assemble(units, [_book_decision(pack[0]["passage_id"])] * len(units))
    with pytest.raises(VerifierValidationError):
        check_exact_quotes(units=units, result=result, passages=pack)


def test_extraction_and_budget_agree_on_every_span() -> None:
    answers = [
        "Без цитат вообще.",
        "Одна «короткая цитата» здесь.",
        "Две «первая» и «вторая цитата» здесь.",
        "Многострочная «первая строка.\nВторая строка» здесь.",
        "Вложенная «внешняя «внутренняя» цитата» здесь.",
        "Немецкая „низкая цитата“ здесь.",
    ]
    for answer in answers:
        extraction = extract_answer_quotes(answer)
        assert not extraction.dangling
        expected = sum(len(span.span_text) for span in extraction.spans if span.span_text.strip())
        assert total_quoted_chars(answer) == expected
        assert aggregate_quote_chars(answer) == expected


def test_nested_quotes_validate_as_whole_spans() -> None:
    book = "Преамбула: внешняя «внутренняя» цитата. Эпилог."
    pack = [_pack_entry(book)]
    draft = "Он сказал: «внешняя «внутренняя» цитата»."
    certificate = _certify(
        draft, [_book_decision(pack[0]["passage_id"])], pack, user_message="что там?"
    )
    assert certificate.passed is True


def test_mismatched_quotes_cannot_be_skipped() -> None:
    draft = "Он сказал: «текст“."
    extraction = extract_answer_quotes(draft)
    assert extraction.dangling
    pack = [_pack_entry("Он сказал: текст.")]
    units = split_response_units(draft)
    result = _assemble(units, [_book_decision(pack[0]["passage_id"])])
    with pytest.raises(VerifierValidationError):
        check_exact_quotes(units=units, result=result, passages=pack)


def test_repaired_text_keeps_accurate_span_offsets() -> None:
    draft = "В книге написано: «Первое предложение. Второе предложение»."
    repaired = draft + " Дополнительное предложение."
    first = extract_answer_quotes(draft).spans[0]
    second = extract_answer_quotes(repaired).spans[0]
    assert (first.answer_char_start, first.answer_char_end) == (
        second.answer_char_start,
        second.answer_char_end,
    )
    assert repaired[first.answer_char_start : first.answer_char_end] == first.span_text


def test_multiunit_quote_mapping_covers_both_units() -> None:
    book = "Преамбула. Первое предложение. Второе предложение. Эпилог."
    pack = [_pack_entry(book)]
    draft = "В книге написано: «Первое предложение. Второе предложение»."
    certificate = _certify(
        draft,
        [_book_decision(pack[0]["passage_id"])] * 2,
        pack,
        user_message="что там?",
    )
    assert certificate.overlap_units == [["u1", "u2"]]
    assert certificate.spans[0].unit_ids == ["u1", "u2"]
    assert certificate.answer_sha256 == _sha(draft)


def test_user_quotes_exempt_from_book_budget_only() -> None:
    user_text = "не могу уснуть вторую ночь"
    draft = f"Вы написали «{user_text}». Что вы имеете в виду?"
    certificate = _certify(
        draft,
        [_user_decision(), _glue_decision()],
        [_pack_entry("Другой текст.")],
        recent=[HumanMessage(content=user_text)],
        user_message=user_text,
    )
    # Total budget still sees the quotation; the book quota exempts it.
    assert aggregate_quote_chars(draft) == len(user_text)
    assert book_quote_chars_for_answer(draft, certificate) == 0
    assert (
        pipeline_book_quotes(
            draft,
            user_span_texts=frozenset([user_text]),
            pack=[],
            recent=[HumanMessage(content=user_text)],
            user_message=user_text,
        )
        == 0
    )
    # Without trusted provenance the same text counts fail closed.
    assert pipeline_book_quotes(draft, user_span_texts=frozenset(), pack=[]) == len(user_text)


# ---------------------------------------------------------------------------
# Compiled-pipeline controls.
# ---------------------------------------------------------------------------


class _StaticAnswer:
    def __init__(self, text: str) -> None:
        self._text = text

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        from langchain_core.messages import AIMessage as _AI

        return _AI(content=self._text)


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


async def test_pipeline_invented_quote_fails_closed() -> None:
    """The full turn fails typed (never serves invented book text)."""
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    book = "Преамбула про поддержку. Первое предложение. Эпилог про трезвость."
    pack = [_pack_entry(book)]
    draft = "В книге написано: «Первое предложение. Второе предложение»."
    verifier = _ScriptedVerifier(
        [
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
            }
        ]
        * 2
    )
    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(
            user_message="Что написано в книге про поддержку?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=_StaticAnswer(draft),
            verifier_model=verifier,
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
        )


async def test_pipeline_exact_quote_is_served() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    exact = "Фиктивная поддержка рядом. Спокойный разговор помогает."
    pack = [_pack_entry(exact + " Эпилог.")]
    draft = f"Вот точные слова: «{exact}»."
    verifier = _ScriptedVerifier(
        [
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
            }
        ]
        * 2
    )
    outcome = await run_v2_answer_turn(
        user_message="Приведи точную цитату про поддержку.",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=_StaticAnswer(draft),
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
    )
    assert exact in outcome["text"]
    assert aggregate_quote_chars(outcome["text"]) <= QUOTE_BUDGET_CHARS
