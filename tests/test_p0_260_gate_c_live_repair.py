"""P0 kodmial/aa#268: Gate C live repair on the model-driven architecture.

The #260 token-level capability classifier and prompt-combining relevance
heuristics are replaced: the hidden planner decides glue vs substantive
through its mode (conversational with zero queries is glue, retrieval is
substantive regardless of wording), and whole-turn relevance comes from
the unified verifier verdicts (per-unit ``addresses_intent`` plus
turn-level ``answer_relevant``). Prior turns were already resolved into
the planner intent, so they are accepted for compatibility and ignored.
The live qualification lane judges relevance with
``assess_reply_relevance_with_rubric`` over authoritative telemetry (or
an injected judge), failing closed without either signal.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Поддержка рядом помогает пережить тягу сегодня.",
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


def _book_grounding(
    passage_id: str = "chapter-3#exp0000",
    unit_text: str = "Поддержка рядом помогает пережить тягу сегодня.",
    addresses_intent: bool = True,
    answer_relevant: bool = True,
) -> dict[str, Any]:
    return {
        "verified": True,
        "all_required_supported": True,
        "answer_relevant": answer_relevant,
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [passage_id],
                "text": unit_text,
                "addresses_intent": addresses_intent,
            }
        ],
    }


def test_conversational_mode_with_zero_queries_is_glue() -> None:
    from aa.conversation.answer_adequacy import is_conversational_plan

    assert is_conversational_plan(mode="conversational", query_count=0) is True
    assert is_conversational_plan(mode="conversational", query_count=4) is False


def test_retrieval_mode_is_substantive_regardless_of_wording() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy, is_conversational_plan

    # Mode alone decides; no turn text is inspected here.
    assert is_conversational_plan(mode="retrieval", query_count=0) is False
    assert is_conversational_plan(mode="retrieval", query_count=12) is False
    for wording in (
        "frozen-260-marker-alpha capability wording",
        "frozen-260-marker-beta capability wording",
    ):
        pack = [_pack_entry()]
        verdict = assess_turn_adequacy(
            user_message=wording,
            reply="Поддержка рядом помогает пережить тягу сегодня.",
            evidence_pack=pack,
            grounding_result=_book_grounding(),
            planner_reason="substantive-with-queries",
            planner_mode="retrieval",
            resolved_intent=wording,
        )
        assert verdict.verdict == "pass"
        assert verdict.substantive_request is True


def test_meta_without_book_passes_adequacy_as_glue() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    verdict = assess_turn_adequacy(
        user_message="frozen-260-marker-alpha capability probe",
        reply="I help with community materials and conversation.",
        evidence_pack=[],
        grounding_result={
            "verified": True,
            "all_required_supported": True,
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "conversation_glue",
                    "supported": True,
                    "evidence_passage_ids": [],
                }
            ],
        },
        planner_reason="legitimate-glue",
        planner_mode="conversational",
    )
    assert verdict.verdict == "pass"
    assert verdict.substantive_request is False


def test_relevance_true_passes_ignoring_prior_messages() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    reply = "Поддержка рядом помогает пережить тягу сегодня."
    pack = [_pack_entry()]
    # Prior turns are accepted for compatibility but must not affect the
    # verdict; the model relevance flags decide.
    for prior in (
        ["frozen-260-marker-alpha prior turn"],
        ["frozen-260-marker-beta prior turn"],
    ):
        verdict = assess_turn_adequacy(
            user_message="frozen-260-marker-alpha follow-up",
            reply=reply,
            evidence_pack=pack,
            grounding_result=_book_grounding(addresses_intent=True, answer_relevant=True),
            planner_reason="substantive-with-queries",
            planner_mode="retrieval",
            resolved_intent="frozen-260-marker-alpha intent",
            prior_user_messages=prior,
        )
        assert verdict.verdict == "pass"
        assert verdict.answers_request is True


def test_relevance_false_fails_ignoring_prior_messages() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    reply = "Поддержка рядом помогает пережить тягу сегодня."
    pack = [_pack_entry()]
    # Same priors as the passing case; only the relevance flags change,
    # so the failure is model-driven rather than context-driven.
    for prior in (
        ["frozen-260-marker-alpha prior turn"],
        ["frozen-260-marker-beta prior turn"],
    ):
        verdict = assess_turn_adequacy(
            user_message="frozen-260-marker-alpha follow-up",
            reply=reply,
            evidence_pack=pack,
            grounding_result=_book_grounding(addresses_intent=False, answer_relevant=False),
            planner_reason="substantive-with-queries",
            planner_mode="retrieval",
            resolved_intent="frozen-260-marker-alpha intent",
            prior_user_messages=prior,
        )
        assert verdict.verdict == "fail"
        assert verdict.failure_category == "irrelevant-citation"


def test_unrelated_citation_still_fails_with_context() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    ev_text = "Financial planning helps keep a calm budget."
    pack = [_pack_entry(passage_id="chapter-9#exp0001", text=ev_text)]
    grounding = {
        "verified": True,
        "all_required_supported": True,
        "answer_relevant": False,
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["chapter-9#exp0001"],
                "text": ev_text,
                "addresses_intent": False,
            }
        ],
    }
    verdict = assess_turn_adequacy(
        user_message="frozen-260-marker-beta follow-up",
        reply=ev_text,
        evidence_pack=pack,
        grounding_result=grounding,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="frozen-260-marker-beta intent",
        prior_user_messages=["frozen-260-marker-alpha prior turn"],
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "irrelevant-citation"


def test_qualification_rubric_uses_telemetry_then_judge() -> None:
    from aa.qualification.product_contract_live import (
        assess_reply_relevance_with_rubric,
    )

    passing_telemetry = {
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
    }
    assert (
        assess_reply_relevance_with_rubric(
            "frozen-260-marker-alpha prompt",
            "frozen-260-marker-alpha reply",
            telemetry=dict(passing_telemetry),
        )
        is True
    )
    failing_telemetry = {
        "adequacy_verdict": "fail",
        "answers_request": False,
        "technically_grounded": False,
    }
    assert (
        assess_reply_relevance_with_rubric(
            "frozen-260-marker-alpha prompt",
            "frozen-260-marker-alpha reply",
            telemetry=dict(failing_telemetry),
        )
        is False
    )
    # No telemetry and no judge fails closed.
    assert (
        assess_reply_relevance_with_rubric(
            "frozen-260-marker-beta prompt", "frozen-260-marker-beta reply"
        )
        is False
    )
    assert (
        assess_reply_relevance_with_rubric(
            "frozen-260-marker-beta prompt",
            "frozen-260-marker-beta reply",
            telemetry=None,
            judge=None,
        )
        is False
    )
    # A stub judge decides only when telemetry is absent.
    assert (
        assess_reply_relevance_with_rubric(
            "frozen-260-marker-gamma prompt",
            "frozen-260-marker-gamma reply",
            judge=lambda prompt, reply, context: True,
        )
        is True
    )
    assert (
        assess_reply_relevance_with_rubric(
            "frozen-260-marker-gamma prompt",
            "frozen-260-marker-gamma reply",
            judge=lambda prompt, reply, context: False,
        )
        is False
    )


def test_no_exact_live_question_special_cases() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "answer_adequacy.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
    ]
    for source in sources:
        for fragment in (
            "frozen-260-marker-alpha",
            "frozen-260-marker-beta",
            "frozen-260-marker-gamma",
        ):
            assert fragment not in source
