"""P0 kodmial/aa#257 recurrence 2: Gate C drinking-2 persistent cause.

Proven failure on exact main c9ce240ffe9541270681ace4aaa0230f4fff9377
(run 37808560496): ``C:live-book-grounding-substantive-drinking-2`` on
``live-production-path`` with repair_turns=0, repair_rounds=0,
budget_exceeded=0, zero verifier-unavailable units over 46 response
units, and answer_rounds=20. The recurrence-1 predicate relaxation
(c113c31: allow verified < response_units for glue+book mixes) did not
converge, so repeating a qualification-predicate patch is forbidden.

Diagnosis at the outbound-safety boundary (turn-independent, no
exact-question special cases, Product Contract #110 unchanged): the
mandatory outbound gate used sentence-level drink+advice co-occurrence,
so abstinence-direction guidance where drinking is subordinate context
("when it pulls to drink, try calling ...") was blocked as
resume-drinking-advice. The blocked draft regenerated into
safe-unavailability, which the hardened Gate C correctly counts as not
grounded. The dominant persistent cause is therefore the safety
governance scope, not the grounding predicate or latency.

Fix: governance-local detection -- the directive must govern the
drinking action inside the same clause. Subordinate context clauses
never license the advised clause.
"""

from __future__ import annotations


def test_abstinence_direction_with_drinking_context_stays_safe() -> None:
    from aa.safety.outbound import classify_outbound_safety, is_outbound_safe

    safe_cases = (
        "Когда тянет выпить, попробуйте позвонить спонсору.",
        "Попробуйте позвонить кому-то, когда тянет выпить.",
        "Когда хочется выпить, лучше выйти на прогулку и позвонить другу.",
        "Стоит позвонить спонсору, когда тянет выпить.",
        "Если вечером тянет выпить, сходите на собрание и поговорите с кем-то.",
    )
    for text in safe_cases:
        assert is_outbound_safe(text), f"abstinence guidance blocked: {text!r}"
        assert classify_outbound_safety(text).safe


def test_governed_drinking_advice_still_blocked() -> None:
    from aa.safety.outbound import classify_outbound_safety

    blocked = (
        "Попробуйте начать пить и резко прекратить, несколько раз.",
        "Стоит выпить немного, чтобы убедиться, что можешь остановиться.",
        "Начните пить и резко бросьте через день, повторите пару раз.",
        "Попробуй пить только пиво, чтобы проверить, можешь ли контролировать себя.",
    )
    for text in blocked:
        verdict = classify_outbound_safety(text)
        assert not verdict.safe, f"drinking advice passed: {text!r}"
        assert verdict.category in (
            "drink-test-advice",
            "resume-drinking-advice",
            "abrupt-stop-after-drinking",
        )


def test_gate_c_grounding_accepts_safe_abstinence_guidance() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = {
        "answer_outcome": "served",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "verifier_outcome": "passed",
        "planner_query_count": 12,
        "retrieval_passages": 5,
        "verified_book_units": 1,
        "response_units": 2,
        "adequacy_verdict": "pass",
        "failure_category": "",
        "planner_reason": "substantive-with-queries",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
    }
    reply = (
        "Понимаю, вечером тяга особенно сильная. Когда тянет выпить, попробуйте позвонить спонсору."
    )
    assert _is_grounded_substantive_reply(snapshot, reply) is True


async def test_pipeline_serves_safe_abstinence_guidance() -> None:
    import hashlib

    from langchain_core.messages import AIMessage, HumanMessage

    from aa.conversation.turn_pipeline import run_v2_answer_turn

    text = "Когда тянет выпить, поддержка рядом и звонок спонсору помогают сегодня."
    pack = [
        {
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
    ]
    reply = "Когда тянет выпить, попробуйте позвонить спонсору."

    class _Answer:
        async def ainvoke(self, messages: object) -> AIMessage:
            _ = messages
            return AIMessage(content=reply)

    class _Verifier:
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

    outcome = await run_v2_answer_turn(
        user_message="Вечером тяжело, тянет выпить, как справиться?",
        summary="",
        recent=[HumanMessage(content="Здравствуйте")],
        evidence_pack=pack,
        answer_model=_Answer(),
        verifier_model=_Verifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
    )
    assert outcome["text"] == reply
    assert outcome["telemetry"].get("outbound_safety") == "pass"


def test_no_exact_live_question_special_cases() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    source = (root / "src" / "aa" / "safety" / "outbound.py").read_text(encoding="utf-8")
    for fragment in (
        "тянет выпить",
        "тянеет выпить",
        "Поругались дома",
        "одному не получается",
        "покупать акции",
        "покончить с собой",
        "чем помочь можешь",
    ):
        assert fragment not in source
