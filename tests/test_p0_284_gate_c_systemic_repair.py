"""P0 kodmial/aa#284 SYSTEMIC (issue #301): generative conversational turns.

A planner-certified conversational turn (model-resolved
``mode == "conversational"`` with zero queries, legitimate-glue reason
and an empty Evidence Pack) whose single draft/verify attempt produces
no verifier verdict is served with one model-composed dialogue reply
(empty evidence, live request verbatim), never a fixed string. The
candidate still passes the envelope, language, leak, quote-budget and
outbound-safety gates. Substantive turns never enter this boundary and
fail as typed outcomes.

Generic coverage only: no literal qualification prompt is used below.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


class _StaticAnswer:
    """Deterministic answer model returning one fixed Russian draft."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        return AIMessage(content=self._text)


class _OutageVerifier:
    """Verifier fake: every unit fails at the transport boundary."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        from aa.opencode.errors import OpenCodeTransientError

        _ = (prompt, system, schema, retry_count)
        raise OpenCodeTransientError("transient outage")

    async def _ainvoke_text(self, text: str, *, system: str | None = None) -> str:
        from aa.opencode.errors import OpenCodeTransientError

        _ = (text, system)
        raise OpenCodeTransientError("transient outage")


class _UnsupportedGlueVerifier:
    """Verifier fake: glue draft judged unsupported (model drift)."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return {
            "requires_book_evidence": False,
            "supported": False,
            "evidence_passage_ids": [],
            "addresses_intent": False,
        }


async def _run_conversational_turn(verifier: Any) -> dict[str, Any]:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    return await run_v2_answer_turn(
        user_message="привет, просто зашёл поздороваться сегодня",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=_StaticAnswer("Я помощник, поддерживаю разговор."),
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
        initial_query_count=0,
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
    )


async def test_conversational_verifier_outage_serves_fallback() -> None:
    from aa.conversation.failures import is_service_error
    from aa.qualification.product_contract_live import _is_direct_meta_reply

    outcome = await _run_conversational_turn(_OutageVerifier())
    # Generative dialogue: the model-composed text is served, never a
    # fixed string and never a marked service error.
    assert outcome["text"] == "Я помощник, поддерживаю разговор."
    assert not is_service_error(outcome["text"])
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "conversational-generated"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["qualified"] is True
    assert _is_direct_meta_reply(
        outcome["text"],
        snapshot={**telemetry, "answer_relevant": True},
    )


async def test_conversational_unsupported_glue_serves_fallback() -> None:
    from aa.conversation.failures import is_service_error

    outcome = await _run_conversational_turn(_UnsupportedGlueVerifier())
    assert outcome["text"] == "Я помощник, поддерживаю разговор."
    assert not is_service_error(outcome["text"])
    assert outcome["telemetry"]["adequacy_verdict"] == "pass"
    assert outcome["telemetry"]["answers_request"] is True


async def test_substantive_verifier_outage_still_fails_closed() -> None:
    import hashlib

    import pytest as _pt

    from aa.conversation.failures import TurnFailed as _TF
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    pack_text = "Книга говорит о поддержке и трезвости сегодня."
    pack = [
        {
            "passage_id": "chapter-3#exp0000",
            "text": pack_text,
            "source_id": "ru-fourth-edition-txt",
            "section_id": "chapter-3",
            "char_start": 0,
            "char_end": len(pack_text),
            "text_sha256": hashlib.sha256(pack_text.encode("utf-8")).hexdigest(),
        }
    ]
    with _pt.raises(_TF) as _exc:
        await run_v2_answer_turn(
            user_message="общий вопрос про поддержку и трезвость сегодня",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=_StaticAnswer("Поддержка рядом помогает сегодня."),
            verifier_model=_OutageVerifier(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
            planner_reason="substantive-with-queries",
            planner_mode="retrieval",
            resolved_intent="общий вопрос про поддержку и трезвость сегодня",
        )
    assert _exc.value.telemetry["adequacy_verdict"] == "fail"


def test_conversational_fallback_text_contract() -> None:
    # Issue #301: no fixed conversational fallback exists in the
    # pipeline; the served dialogue text is model-composed and still
    # passes all output gates.
    import pathlib as _pl

    from aa.conversation.output_limits import aggregate_quote_chars, envelope_passes
    from aa.conversation.turn_pipeline import contains_cyrillic, leaks_internal_terms
    from aa.safety.outbound import is_outbound_safe

    _src = (
        _pl.Path(__file__).resolve().parents[1] / "src" / "aa" / "conversation" / "turn_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "CONVERSATIONAL_FALLBACK_REPLY" not in _src
    text = "Я помощник, поддерживаю разговор."
    assert contains_cyrillic(text)
    assert not leaks_internal_terms(text)
    assert envelope_passes(text)
    assert aggregate_quote_chars(text) == 0
    assert is_outbound_safe(text)
    lowered = text.casefold()
    for marker in ("я человек", "я спонсор", "я врач", "лет трезвости"):
        assert marker not in lowered


def test_no_exact_question_branches() -> None:
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "conversation"
        / "turn_pipeline.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "Чем ты вообще можешь быть полезен",
        "Слушай, а ты тут вообще чем помочь можешь",
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
        "Дома снова ссора",
        "Поругались дома из-за бухла",
        "покупать акции",
        "Какой телефон сейчас выгоднее купить",
    ):
        assert fragment not in source
