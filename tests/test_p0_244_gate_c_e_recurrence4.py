"""P0 kodmial/aa#244 recurrence 4: support-vocabulary relevance bridge.

Proven product failures on exact main
5b3fd771ef2586ee2be6140f1040b4bafe2c3b52 (run 37819349221):

- C:live-book-grounding-substantive-drinking-2 on live-production-path;
- E:latency-budget-exceeded on slo (p50 23071ms / p95 60226ms /
  max 76026ms with planner p50 6077ms / p95 10376ms, retrieval p50
  428ms / p95 993ms, answer p50 8479ms / p95 15691ms, verifier p50
  8350ms / p95 22862ms / max 46582ms, repair_turns=1, repair_rounds=1,
  answer_rounds=24, budget_exceeded=0, unavailable 0/0 over 48
  response units with max 4, message-text p50 4662ms / p95 11170ms /
  max 31163ms over 141 slow calls vs message-structured p50 410ms /
  p95 718ms over 45 fast calls).

Compared with the recurrence-3 base (exact main 1d109a2 run
37771443722: p50 19.5s / p95 27.0s, text 79 calls, verifier p95 12.0s),
the persistent-rejection pinning did not converge: call counts
exploded (+62 text, +12 structured) and every stage worsened, with
the verifier tail doubling. The waste is no longer attempt duration
(structured serves fast at p50 0.41s) but the second sequential model
sequence burned after an already-slow initial chain. Prior strategies
(safety predicate scope in #257, anchored repair focus plus repair
budget in #269, capability-cache streak in recurrence 3) never
touched this lexical gap, so repeating them cannot converge.

Dominant persistent cause at the adequacy/relevance boundary: good
abstinence-direction guidance uses support vocabulary (sponsor,
meeting, fellowship, prayer, community, recovery) without repeating
craving words, so it shares no prefix overlap and no recovery-domain
token with the request. Both the adequacy domain fallback and the
live relevance domain check therefore score it exactly like an
unrelated finance fact: the turn fails C (reported first as grounding
when both fail) and the adequacy-regeneration answer+verifier round
fires, still fails on the same lexical gap, and breaches the E p95
tail. Strategy change here (not another budget/TTL/capability
retune): a generic support-domain bridge lets an evidence-backed
support unit count as relevant to a recovery-domain request. Passing
on the initial draft also skips the regen sequence, cutting the tail
for Gate E with the same mechanism. Generic stems only, never an
exact-question list; finance/all-glue/wrong-step negatives still
fail. Turn-independent, Product Contract #110 unchanged, no
SLO/threshold weakening.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Support nearby helps to get through craving today.",
) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def test_budgets_thresholds_and_slo_unchanged() -> None:
    """Recurrence 4 changes relevance semantics only; walls and SLO stay strict."""
    from aa.conversation.planner_node import (
        PLANNER_STRUCTURED_ATTEMPT_BUDGET_S,
        PLANNER_TIME_BUDGET_S,
    )
    from aa.conversation.turn_pipeline import (
        ANSWER_DRAFT_ATTEMPT_BUDGET_S,
        TURN_END_TO_END_BUDGET_S,
        TURN_REPAIR_TIME_BUDGET_S,
    )
    from aa.conversation.verifier import (
        VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S,
        VERIFIER_TURN_BUDGET_S,
    )
    from aa.qualification.self_proving import ORDINARY_TURN_BUDGET_MS, P95_TARGET_MS

    assert PLANNER_STRUCTURED_ATTEMPT_BUDGET_S == 6.0
    assert PLANNER_TIME_BUDGET_S == 25.0
    assert VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S == 6.0
    assert VERIFIER_TURN_BUDGET_S == 40.0
    assert ANSWER_DRAFT_ATTEMPT_BUDGET_S == 35.0
    assert TURN_REPAIR_TIME_BUDGET_S == 50.0
    assert TURN_END_TO_END_BUDGET_S == 105.0
    assert P95_TARGET_MS == 60_000
    assert ORDINARY_TURN_BUDGET_MS == 120_000


def test_support_vocabulary_counts_as_relevant_to_recovery_request() -> None:
    """A support-step reply with no drinking words is relevant (not finance)."""
    from aa.qualification.product_contract_live import _assess_prompt_reply_relevance

    prompt = "Вечером тяжело пережить тягу, как обходиться без спиртного?"
    support = (
        "Позвоните спонсору и сходите на собрание, поддержка рядом помогает пережить этот вечер."
    )
    assert _assess_prompt_reply_relevance(prompt, support) is True


def test_negative_controls_still_fail() -> None:
    """Finance, wrong-step and all-glue negatives are preserved."""
    from aa.qualification.product_contract_live import _assess_prompt_reply_relevance

    assert (
        _assess_prompt_reply_relevance(
            "Вечером тяжело пережить тягу",
            "Ведите финансовый бюджет спокойно.",
        )
        is False
    )
    assert (
        _assess_prompt_reply_relevance(
            "Расскажи про Первый шаг",
            "Третий шаг говорит о решениях и воле.",
        )
        is False
    )
    # A support reply to a finance request stays irrelevant (bridge needs
    # a recovery-domain request, so stale context cannot rescue a pivot).
    assert (
        _assess_prompt_reply_relevance(
            "Стоит ли мне сейчас покупать акции?",
            "Позвоните спонсору и сходите на собрание.",
        )
        is False
    )


def test_adequacy_passes_evidence_backed_support_unit() -> None:
    """Adequacy accepts a support unit backed by a support passage."""
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    prompt = "Вечером тяжело пережить тягу, как обходиться без спиртного?"
    reply = (
        "Позвоните спонсору и сходите на собрание, поддержка рядом помогает пережить этот вечер."
    )
    pack = [
        _pack_entry(
            "chapter-3#exp0000",
            "Позвоните спонсору и приходите на собрание: поддержка сообщества "
            "помогает оставаться трезвым сегодня.",
        )
    ]
    grounding = {
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "text": "Позвоните спонсору и приходите на собрание.",
            },
            {
                "unit_id": "u2",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "text": "Поддержка сообщества помогает оставаться трезвым сегодня.",
            },
        ],
        "all_required_supported": True,
    }
    assessment = assess_turn_adequacy(
        user_message=prompt,
        reply=reply,
        evidence_pack=pack,
        grounding_result=grounding,
        planner_reason="substantive-with-queries",
        verifier_outcome="passed",
        unavailable_units=0,
        turn_budget_exceeded=False,
        summary="",
        recent=[],
        resolved_request=prompt,
        prior_user_messages=[],
    )
    assert assessment.verdict == "pass"
    assert assessment.answers_request is True


def test_adequacy_still_fails_unrelated_citation() -> None:
    """A finance unit for a craving request still fails adequacy."""
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    prompt = "Вечером тяжело пережить тягу, как обходиться без спиртного?"
    reply = "Ведите финансовый бюджет спокойно."
    pack = [_pack_entry("chapter-9#exp0000", "Ведите финансовый бюджет спокойно.")]
    grounding = {
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["chapter-9#exp0000"],
                "text": "Ведите финансовый бюджет спокойно.",
            }
        ],
        "all_required_supported": True,
    }
    assessment = assess_turn_adequacy(
        user_message=prompt,
        reply=reply,
        evidence_pack=pack,
        grounding_result=grounding,
        planner_reason="substantive-with-queries",
        verifier_outcome="passed",
        unavailable_units=0,
        turn_budget_exceeded=False,
        summary="",
        recent=[],
        resolved_request=prompt,
        prior_user_messages=[],
    )
    assert assessment.verdict == "fail"
    assert assessment.answers_request is False


def test_grounded_gate_accepts_support_guidance_with_pass_snapshot() -> None:
    """Gate C grounding holds for support guidance once adequacy passes."""
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = {
        "answer_outcome": "served",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "verifier_outcome": "passed",
        "planner_query_count": 12,
        "retrieval_passages": 5,
        "verified_book_units": 2,
        "response_units": 3,
        "adequacy_verdict": "pass",
        "failure_category": "",
        "planner_reason": "substantive-with-queries",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
    }
    reply = (
        "Позвоните спонсору и сходите на собрание, поддержка рядом помогает пережить этот вечер."
    )
    assert _is_grounded_substantive_reply(snapshot, reply) is True


async def test_pipeline_serves_support_guidance_in_one_sequence() -> None:
    """The initial draft passes adequacy, so no regen sequence burns (Gate E).

    One answer call plus one verifier round serves the turn; the second
    answer+verifier chain that breached the p95 tail never starts.
    """
    from aa.conversation import turn_pipeline as pipeline

    support_text = (
        "Позвоните спонсору и приходите на собрание: поддержка сообщества "
        "помогает оставаться трезвым сегодня."
    )
    pack = [_pack_entry("chapter-3#exp0000", support_text)]
    reply = (
        "Позвоните спонсору и сходите на собрание, поддержка рядом помогает пережить этот вечер."
    )

    class _Answer:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: object) -> AIMessage:
            _ = messages
            self.calls += 1
            return AIMessage(content=reply)

    class _Verifier:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
            }

    answer = _Answer()
    outcome = await pipeline.run_v2_answer_turn(
        user_message="Вечером тяжело пережить тягу, как обходиться без спиртного?",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=_Verifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
    )
    assert outcome["text"] == reply
    telemetry = outcome["telemetry"]
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answer_outcome"] == "served"
    # Single answer sequence: no adequacy-regen second call.
    assert answer.calls == 1
    assert telemetry["answer_rounds"] == 1
    assert telemetry["repair_rounds"] == 0


def test_outbound_safety_contract_preserved() -> None:
    """True drink-test advice still blocks; abstinence guidance stays safe."""
    from aa.safety.outbound import is_outbound_safe

    assert is_outbound_safe("Попробуйте позвонить спонсору, когда тянет выпить.") is True
    assert is_outbound_safe("Попробуйте начать пить и резко прекратить, несколько раз.") is False


def test_no_exact_live_question_special_cases() -> None:
    """The repair stays turn-independent: no live prompt text in product sources.

    The qualification lane itself defines the frozen scenarios, so it is
    excluded here exactly like the recurrence-3 lock (which scopes this
    check to the conversation transport files).
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "answer_adequacy.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "model_adapter.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "prompt_builder.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
    ]
    frozen_fragments = (
        "тянет выпить",
        "тянеет выпить",
        "ссора из-за моей выпивки",
        "Поругались дома",
        "не могу успокоиться и уснуть",
        "мысли крутятся",
        "покупать акции",
        "выгоднее купить",
        "покончить с собой",
        "Не хочу жить",
        "одному не получается",
        "чем помочь можешь",
    )
    for source in sources:
        for fragment in frozen_fragments:
            assert fragment not in source
