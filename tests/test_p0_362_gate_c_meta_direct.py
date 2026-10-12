"""P0 kodmial/aa#362: Gate C meta-direct fallback telemetry must stay certified.

Proven product failure on exact main dece957 (run 38234180963):

- C:live-meta-direct-1:live-production-path (plus D:workflow-step-failed
  downstream and E:latency-budget-exceeded SLO).

Live diagnostics show the functional defect is in the conversational
fallback path, not only latency: the first meta-capability turn collapsed
to live-meta-direct failure while clarifications=10 and two turns carried
unavailable verifier units with two budget-exceeded turns.

Root cause (failure class, not one prompt): two planner-certified glue
delivery paths served a model-composed dialogue reply without the
certified conversational telemetry. The initial-draft-missing path
(answer timeout/budget/transient with proven glue) and the
repair-exhausted tail path (proven glue with no book claim) both
returned ``conversational-generated`` with the initial unknown/failed
adequacy (adequacy unknown, answers False, qualified False). The live
lane judges the sent text plus this turn's snapshot via
``_is_direct_meta_reply``, which requires adequacy pass plus answers
True (or relevant True) and fails closed otherwise. A perfectly direct
natural meta reply was therefore counted as a production-path failure
instead of a direct answer.

Repair (generic, no exact-question special cases; Product Contract #110
unchanged): both glue fallbacks certify the same conversational delivery
telemetry as the main fallback (outbound pass, conversational-generated,
adequacy pass, answers True, technically_grounded False, qualified
True). The turn still serves exactly one model-composed dialogue reply
through the same envelope/language/leak/quote/safety gates, never
synthetic substantive conversation.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


class _StaticAnswer:
    def __init__(self, text: str) -> None:
        self._text = text

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        return AIMessage(content=self._text)


class _FailOnceThenSucceed:
    """First answer attempt fails, dialogue fallback succeeds."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        _ = messages
        if self.calls == 1:
            raise RuntimeError("fixture transient answer failure")
        return AIMessage(content=self._text)


class _OutageVerifier:
    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        from aa.opencode.errors import OpenCodeTransientError

        _ = (prompt, system, schema, retry_count)
        raise OpenCodeTransientError("fixture transient outage")

    async def _ainvoke_text(self, text: str, *, system: str | None = None) -> str:
        from aa.opencode.errors import OpenCodeTransientError

        _ = (text, system)
        raise OpenCodeTransientError("fixture transient outage")


def _pack_entry() -> dict[str, Any]:
    import hashlib

    text = "Изобретение спокойного вечернего распорядка помогает сегодня."
    return {
        "passage_id": "chapter-3#exp0000",
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


async def test_draft_missing_glue_still_certifies_direct_meta() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import _is_direct_meta_reply

    dialogue = "Я помощник, поддерживаю разговор о трезвости сегодня."
    outcome = await run_v2_answer_turn(
        user_message="Нейтральное приветствие и вопрос о возможностях сегодня",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=_FailOnceThenSucceed(dialogue),
        verifier_model=_OutageVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=0,
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
    )
    assert outcome["text"] == dialogue
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "conversational-generated"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["qualified"] is True
    assert telemetry["technically_grounded"] is False
    assert _is_direct_meta_reply(outcome["text"], snapshot=dict(telemetry)) is True


async def test_repair_tail_glue_still_certifies_direct_meta() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import _is_direct_meta_reply

    dialogue = "Я помощник, поддерживаю разговор о трезвости сегодня."
    outcome = await run_v2_answer_turn(
        user_message="Нейтральное приветствие и вопрос о возможностях сегодня",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_StaticAnswer(dialogue),
        verifier_model=_OutageVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=0,
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
    )
    # Main fallback requires an empty pack, so a stale non-empty pack for a
    # proven-glue turn must still certify via the repair-exhausted tail.
    assert outcome["text"] == dialogue
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "conversational-generated"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["qualified"] is True
    assert _is_direct_meta_reply(outcome["text"], snapshot=dict(telemetry)) is True


def test_no_exact_question_special_cases() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    source = (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8")
    for fragment in (
        "Чем ты вообще можешь быть полезен",
        "Слушай, а ты тут вообще чем помочь можешь",
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
        "Дома снова ссора",
        "Поругались дома из-за бухла",
        "покупать акции",
        "Какой телефон сейчас выгоднее купить",
        "Изобретение спокойного вечернего распорядка",
    ):
        assert fragment not in source
