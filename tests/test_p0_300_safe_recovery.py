"""P0 kodmial/aa#300: intent-preserving safe recovery after an outbound veto.

Authored synthetic fixtures only (never real Telegram input/output, never
canonical excerpts). Covers: unsafe draft plus same-pack safe verified
regeneration, zero novel passages, additional retrieval success,
false-positive veto, missing index/provider, time-budget exhaustion,
explicit drink-self-test request, dangerous detox advice, stable
telemetry plus Gate C FAIL for any unhelpful fallback, and no-leak
checks for the separate safety control channel.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aa.safety.outbound import (
    SAFE_RECOVERY_INSTRUCTION,
    SAFE_UNAVAILABLE_REPLY,
    is_outbound_safe,
)

_HARMFUL_DRINK_TEST = "Попробуйте начать пить и резко прекратить, повторите несколько раз."
_HARMFUL_DETOX = (
    "Для снятия абстиненции принимайте спиртное по расписанию, пейте меньше каждый день."
)
_SAFE_HELP_1 = (
    "Поддержка рядом помогает пережить тягу сегодня. Расскажите, что сейчас важнее всего?"
)
_SAFE_HELP_2 = "Честный разговор о трудностях и ближайшие трезвые шаги помогают сегодня."
_SAFE_HELP_3 = "Спокойный вечерний распорядок и поддержка рядом помогают сегодня."


def _pack_entry(passage_id: str = "chapter-3#exp0000", text: str = _SAFE_HELP_3) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class _ScriptedAnswer:
    """Return scripted drafts in order, recording generation prompts."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0
        self.seen_user_messages: list[str] = []
        self.seen_safety_blocks = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        from langchain_core.messages import BaseMessage

        self.calls += 1
        if isinstance(messages, list):
            for item in messages:
                if isinstance(item, BaseMessage) and item.type == "human":
                    content = item.content
                    if isinstance(content, str):
                        if "<user_message>" in content:
                            self.seen_user_messages.append(content)
                        if "<safety_policy>" in content:
                            self.seen_safety_blocks += 1
        index = min(self.calls - 1, len(self._texts) - 1)
        return AIMessage(content=self._texts[index])


class _SupportingVerifier:
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


def _is_gate_c_fail(telemetry: dict[str, Any], text: str) -> bool:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    return _is_grounded_substantive_reply(dict(telemetry), text) is False


async def test_same_pack_safe_regeneration_serves_helpful_answer() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    outcome = await run_v2_answer_turn(
        user_message="Вечером тяжело без выпивки, как справляться?",
        summary="",
        recent=[HumanMessage(content="Здравствуйте")],
        evidence_pack=[_pack_entry()],
        answer_model=_ScriptedAnswer([_HARMFUL_DRINK_TEST, _SAFE_HELP_1]),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
        planner_mode="retrieval",
        resolved_intent="Вечером тяжело без выпивки, как справляться?",
    )
    assert outcome["text"] == _SAFE_HELP_1
    assert is_outbound_safe(outcome["text"])
    assert _HARMFUL_DRINK_TEST not in outcome["text"]
    telemetry = dict(outcome["telemetry"])
    assert telemetry.get("outbound_safety") == "repaired"
    assert telemetry.get("answer_outcome") == "served"
    assert telemetry.get("adequacy_verdict") == "pass"
    assert telemetry.get("qualified") is True


async def test_zero_novel_passages_still_regenerate_from_existing_pack() -> None:
    # No planner/index at all: recovery must still try the existing pack
    # instead of equating "no novel evidence" with "no safe response".
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    outcome = await run_v2_answer_turn(
        user_message="Подскажите, как удержаться сегодня?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_ScriptedAnswer([_HARMFUL_DRINK_TEST, _SAFE_HELP_2]),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
        planner_mode="retrieval",
        resolved_intent="Подскажите, как удержаться сегодня?",
    )
    assert outcome["text"] == _SAFE_HELP_2
    assert outcome["telemetry"].get("answer_outcome") == "served"


async def test_additional_retrieval_success_targets_genuine_need(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.conversation.planner_node as planner_node
    import aa.retrieval.evidence as evidence_mod
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    seen_queries: list[str] = []

    class _Planner:
        pass

    async def _fake_run_planner(
        focus: str, *, model: Any = None, summary: str = "", recent: Any = None
    ) -> Any:
        from aa.conversation.planner_schema import QueryPlan

        _ = (model, summary, recent)
        seen_queries.append(focus)
        return QueryPlan(
            mode="retrieval",
            resolved_intent="Подскажите, как удержаться сегодня?",
            queries=["Подскажите, как удержаться сегодня?"],
        )

    def _fake_retrieve(index: Any, queries: Any, *, config: Any = None, **kwargs: Any) -> Any:
        _ = (index, config, kwargs)
        seen_queries.extend(list(queries))
        pack_text = _SAFE_HELP_3

        from aa.retrieval.evidence import EvidencePack, EvidencePassageData

        passage = EvidencePassageData(
            passage_id="chapter-3#exp0001",
            exact_text=pack_text,
            source_id="ru-fourth-edition-txt",
            section_id="chapter-3",
            child_chunk_ids=("chapter-3#exp0001",),
            char_start=0,
            char_end=len(pack_text),
            text_sha256=hashlib.sha256(pack_text.encode("utf-8")).hexdigest(),
            source_sha256=hashlib.sha256(b"ru-fourth-edition-txt").hexdigest(),
        )
        return EvidencePack(
            passages=(passage,),
            total_tokens=10,
            corpus_version="test",
            retrieval_metadata={},
        )

    monkeypatch.setattr(planner_node, "run_planner", _fake_run_planner)
    monkeypatch.setattr(evidence_mod, "retrieve_evidence", _fake_retrieve)
    outcome = await run_v2_answer_turn(
        user_message="Подскажите, как удержаться сегодня?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_ScriptedAnswer([_HARMFUL_DRINK_TEST, _HARMFUL_DRINK_TEST, _SAFE_HELP_3]),
        verifier_model=_SupportingVerifier(),
        planner_model=_Planner(),
        retrieval_index=object(),
        initial_query_count=12,
        upstream_latency_ms=0.0,
        planner_mode="retrieval",
        resolved_intent="Подскажите, как удержаться сегодня?",
    )
    assert outcome["text"] == _SAFE_HELP_3
    assert outcome["telemetry"].get("answer_outcome") == "served"
    # Retrieval targeted the genuine need, never the safety instruction.
    joined = " ".join(seen_queries)
    assert SAFE_RECOVERY_INSTRUCTION not in joined
    assert "Никогда не советуй" not in joined


async def test_false_positive_veto_ordinary_help_passes_gate() -> None:
    # Ordinary abstinence help must never be vetoed in the first place.
    assert is_outbound_safe(_SAFE_HELP_1)
    assert is_outbound_safe(_SAFE_HELP_2)
    assert is_outbound_safe(_SAFE_HELP_3)
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    outcome = await run_v2_answer_turn(
        user_message="Тянет вечером, что помогает оставаться трезвым?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_ScriptedAnswer([_SAFE_HELP_1]),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
        planner_mode="retrieval",
        resolved_intent="Тянет вечером, что помогает оставаться трезвым?",
    )
    assert outcome["text"] == _SAFE_HELP_1
    assert outcome["telemetry"].get("outbound_safety") == "pass"


async def test_missing_index_provider_fails_closed_neutral() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    outcome = await run_v2_answer_turn(
        user_message="Вечером тяжело, как справляться без выпивки?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_ScriptedAnswer([_HARMFUL_DRINK_TEST, _HARMFUL_DRINK_TEST]),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
        planner_mode="retrieval",
        resolved_intent="Вечером тяжело, как справляться без выпивки?",
    )
    assert outcome["text"] == SAFE_UNAVAILABLE_REPLY
    telemetry = dict(outcome["telemetry"])
    assert telemetry.get("answer_outcome") == "safety-blocked"
    assert telemetry.get("outbound_safety") == "blocked"
    assert telemetry.get("adequacy_verdict") == "fail"
    assert telemetry.get("qualified") is False
    assert _is_gate_c_fail(telemetry, outcome["text"])
    lowered = outcome["text"].casefold()
    assert "пробовать пить" not in lowered
    assert "проверить себя" not in lowered
    assert "пить" not in lowered


async def test_time_budget_exhaustion_fails_closed_neutral() -> None:
    from aa.conversation.turn_pipeline import (
        NATURAL_RETRY_REPLY,
        run_v2_answer_turn,
    )

    outcome = await run_v2_answer_turn(
        user_message="Вечером тяжело, как справляться без выпивки?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_ScriptedAnswer([_HARMFUL_DRINK_TEST]),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=10_000_000.0,
        planner_mode="retrieval",
        resolved_intent="Вечером тяжело, как справляться без выпивки?",
    )
    # Budget exhaustion before any draft fails closed to a neutral retry
    # (initial attempt) or the neutral safety fallback; both are
    # context-independent, non-success, and Gate C FAIL.
    assert outcome["text"] in (SAFE_UNAVAILABLE_REPLY, NATURAL_RETRY_REPLY)
    lowered = outcome["text"].casefold()
    assert "пробовать пить" not in lowered
    assert "проверить себя" not in lowered
    assert "пить" not in lowered
    telemetry = dict(outcome["telemetry"])
    assert telemetry.get("turn_budget_exceeded") is True
    assert telemetry.get("qualified") is False
    assert _is_gate_c_fail(telemetry, outcome["text"])


async def test_explicit_drink_self_test_request_never_serves_harm() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    outcome = await run_v2_answer_turn(
        user_message="Можно ли мне проверить себя, выпив немного?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_ScriptedAnswer([_HARMFUL_DRINK_TEST, _HARMFUL_DRINK_TEST]),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
        planner_mode="retrieval",
        resolved_intent="Можно ли мне проверить себя, выпив немного?",
    )
    assert _HARMFUL_DRINK_TEST not in outcome["text"]
    assert is_outbound_safe(outcome["text"])
    assert outcome["text"] == SAFE_UNAVAILABLE_REPLY
    telemetry = dict(outcome["telemetry"])
    assert telemetry.get("answer_outcome") == "safety-blocked"
    assert _is_gate_c_fail(telemetry, outcome["text"])


async def test_dangerous_detox_advice_is_blocked() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    outcome = await run_v2_answer_turn(
        user_message="Как пережить отмену сегодня?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_ScriptedAnswer([_HARMFUL_DETOX, _HARMFUL_DETOX]),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
        planner_mode="retrieval",
        resolved_intent="Как пережить отмену сегодня?",
    )
    assert _HARMFUL_DETOX not in outcome["text"]
    assert is_outbound_safe(outcome["text"])
    assert outcome["text"] == SAFE_UNAVAILABLE_REPLY
    assert _is_gate_c_fail(dict(outcome["telemetry"]), outcome["text"])


async def test_follow_up_retains_context_and_serves_grounded_help() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    prior_user = HumanMessage(content="Вечером тяжело без выпивки, как справляться?")
    prior_ai = AIMessage(content=_SAFE_HELP_1)
    outcome = await run_v2_answer_turn(
        user_message="А если тяга вернётся позже?",
        summary="Обсуждали вечернюю тягу и поддержку рядом.",
        recent=[prior_user, prior_ai],
        evidence_pack=[_pack_entry()],
        answer_model=_ScriptedAnswer([_SAFE_HELP_2]),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=8,
        upstream_latency_ms=0.0,
        planner_mode="retrieval",
        resolved_intent="Что делать, если вечерняя тяга вернётся позже?",
    )
    assert outcome["text"] == _SAFE_HELP_2
    assert outcome["telemetry"].get("answer_outcome") == "served"


def test_safety_policy_stays_separate_from_user_message_and_queries() -> None:
    from aa.conversation.prompt_builder import render_turn_context
    from aa.conversation.turn_pipeline import (
        build_safety_recovery_queries,
        safety_recovery_request,
    )

    resolved = safety_recovery_request(
        resolved_intent="Вечером тяжело, как справляться?",
        user_message="Вечером тяжело, как справляться?",
    )
    assert SAFE_RECOVERY_INSTRUCTION not in resolved
    queries = build_safety_recovery_queries(
        resolved, summary="", recent_texts=["здравствуйте"], max_queries=6
    )
    assert queries
    assert all(SAFE_RECOVERY_INSTRUCTION not in query for query in queries)
    assert all("Никогда не советуй" not in query for query in queries)
    rendered = render_turn_context(
        summary="",
        passages=[],
        user_message=resolved,
        safety_policy=SAFE_RECOVERY_INSTRUCTION,
    )
    assert "<safety_policy>" in rendered
    assert "<user_message>" in rendered
    user_block = rendered.split("<user_message>")[1].split("</user_message>")[0]
    assert SAFE_RECOVERY_INSTRUCTION not in user_block


def test_recovery_generation_uses_separate_safety_channel() -> None:
    import asyncio

    from aa.conversation.turn_pipeline import run_v2_answer_turn

    model = _ScriptedAnswer([_HARMFUL_DRINK_TEST, _SAFE_HELP_1])
    outcome = asyncio.run(
        run_v2_answer_turn(
            user_message="Вечером тяжело без выпивки, как справляться?",
            summary="",
            recent=[],
            evidence_pack=[_pack_entry()],
            answer_model=model,
            verifier_model=_SupportingVerifier(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
            upstream_latency_ms=0.0,
            planner_mode="retrieval",
            resolved_intent="Вечером тяжело без выпивки, как справляться?",
        )
    )
    assert outcome["text"] == _SAFE_HELP_1
    # At least one recovery generation carried the separate control block,
    # while no retrieval/user payload merged it into the request text.
    assert model.seen_safety_blocks >= 1
    for payload in model.seen_user_messages:
        user_block = payload.split("<user_message>")[1].split("</user_message>")[0]
        assert SAFE_RECOVERY_INSTRUCTION not in user_block


def test_no_domain_keyword_or_exact_utterance_routing() -> None:
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "conversation"
        / "turn_pipeline.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "Вечером тяжело без выпивки",
        "Можно ли мне проверить себя",
        "Как пережить отмену",
        "Спокойный вечерний распорядок",
    ):
        assert fragment not in source
    # Recovery helpers must not consult domain stems.
    assert "тяг" not in source.split("def safety_recovery_request")[1].split("def ")[0]


def test_fallback_telemetry_stable_and_gate_c_fails() -> None:
    import asyncio

    from aa.conversation.turn_pipeline import run_v2_answer_turn

    outcome = asyncio.run(
        run_v2_answer_turn(
            user_message="Подскажите, как удержаться сегодня?",
            summary="",
            recent=[],
            evidence_pack=[_pack_entry()],
            answer_model=_ScriptedAnswer([_HARMFUL_DRINK_TEST, _HARMFUL_DRINK_TEST]),
            verifier_model=_SupportingVerifier(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
            upstream_latency_ms=0.0,
            planner_mode="retrieval",
            resolved_intent="Подскажите, как удержаться сегодня?",
        )
    )
    telemetry = dict(outcome["telemetry"])
    assert telemetry.get("outbound_safety") == "blocked"
    assert isinstance(telemetry.get("outbound_safety_category"), str)
    assert telemetry.get("outbound_safety_category")
    assert telemetry.get("answer_outcome") == "safety-blocked"
    assert telemetry.get("adequacy_verdict") == "fail"
    assert telemetry.get("answers_request") is False
    assert telemetry.get("technically_grounded") is False
    assert telemetry.get("qualified") is False
    assert _is_gate_c_fail(telemetry, outcome["text"])
