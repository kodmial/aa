"""AUDIT P0-3 (kodmial/aa#305): one canonical context + resolved intent.

Regressions: topic return beyond the 12-message window, trailing
condition in a long message, pronoun/referent conflicts, topic
switch/return, user correction, identical stage digests, and summary
never becoming book evidence.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def _history_with_old_topic() -> list[Any]:
    history: list[Any] = [HumanMessage(content="alpha-topic-marker про тягу утром")]
    history.append(AIMessage(content="понимаю, рассказывайте"))
    for index in range(14):
        history.append(HumanMessage(content=f"болтовня filler {index}"))
        history.append(AIMessage(content=f"ответ filler {index}"))
    return history


def test_canonical_context_built_once_with_digest() -> None:
    from aa.conversation.conversation_context import (
        build_conversation_context,
        context_digest_for_canonical,
    )

    history = _history_with_old_topic()
    context = build_conversation_context(
        history, "краткое резюме", "вернёмся к alpha-topic-marker?"
    )
    assert context.user_message == "вернёмся к alpha-topic-marker?"
    assert len(context.messages) == len(history)
    assert context.summary == "краткое резюме"
    assert context.summary_provenance == "continuity-only-not-evidence"
    roles = {item.role for item in context.messages}
    assert {"user", "assistant"} <= roles
    digest = context_digest_for_canonical(context)
    assert len(digest) == 64
    assert context_digest_for_canonical(context.model_dump(mode="json")) == digest


def test_topic_return_beyond_12_window_reaches_all_stages() -> None:

    from aa.conversation.conversation_context import (
        build_conversation_context,
        canonical_model_view,
        select_relevant_history,
    )
    from aa.conversation.planner_node import build_planner_messages
    from aa.conversation.prompt_builder import build_answer_messages
    from aa.conversation.response_units import ResponseUnitDraft
    from aa.conversation.semantic_selection import selection_prompt
    from aa.conversation.verifier import build_single_unit_text

    history = _history_with_old_topic()
    live = "вернёмся к alpha-topic-marker, что делать?"
    context = build_conversation_context(history, "", live)
    dumped = context.model_dump(mode="json")

    selected = select_relevant_history(dumped, resolved_intent=live, max_messages=12)
    assert any("alpha-topic-marker" in item.text for item in selected)

    planner = build_planner_messages(
        user_message=live, summary="", recent=[], conversation_context=dumped
    )
    assert "alpha-topic-marker" in str(planner[1].content)

    view = canonical_model_view(dumped)
    _, selector_text = selection_prompt(
        previews=[],
        resolved_intent=live,
        conversation_context=view["combined"],
        user_message=live,
    )
    assert "alpha-topic-marker" in selector_text

    generator = build_answer_messages(
        recent=list(history),
        summary="",
        passages=[],
        user_message=live,
        resolved_intent=live,
        conversation_context=dumped,
    )
    assert "alpha-topic-marker" in str(generator[-1].content) or "alpha-topic-marker" in str(
        generator[1].content
    )

    verifier_text = build_single_unit_text(
        unit=ResponseUnitDraft(unit_id="u1", text="Короткий ответ.", char_start=0, char_end=5),
        passages=[],
        resolved_intent=live,
        user_message=live,
        conversation_context=view["combined"],
    )
    assert "alpha-topic-marker" in verifier_text
    # Legacy most-recent-12 slicing alone would drop the old topic.
    legacy = history[-12:]
    assert not any("alpha-topic-marker" in str(getattr(item, "content", "")) for item in legacy)


def test_trailing_condition_survives_shared_truncation() -> None:
    from aa.conversation.conversation_context import truncate_preserving_tail

    head = "вступление " * 400
    tail_condition = "ВАЖНОЕ-УСЛОВИЕ-хвост только вечером, никогда утром"
    long_text = head + tail_condition
    assert len(long_text) > 2000
    shared = truncate_preserving_tail(long_text, 2000)
    assert tail_condition in shared
    assert "truncated" in shared
    # Naive first-N clipping would drop the trailing condition.
    assert tail_condition not in long_text[:2000]

    from aa.conversation.conversation_context import (
        build_conversation_context,
        canonical_model_view,
    )
    from aa.conversation.planner_node import build_planner_messages

    history = [HumanMessage(content=long_text)]
    context = build_conversation_context([], "", "что делать?")
    # Full canonical text is preserved, not clipped.
    full = build_conversation_context(history, "", "что делать?")
    assert tail_condition in full.messages[0].text
    dumped = full.model_dump(mode="json")
    view = canonical_model_view(dumped)
    _ = (context, view)
    planner = build_planner_messages(
        user_message="что делать?",
        summary="",
        recent=list(history),
        conversation_context=None,
    )
    # Planner uses the shared tail-preserving bound, so the tail survives
    # even through the bounded display.
    assert tail_condition in str(planner[1].content)


def test_correction_and_referents_preserve_roles() -> None:
    from aa.conversation.conversation_context import build_conversation_context

    history = [
        HumanMessage(content="я говорил про утреннюю тягу"),
        AIMessage(content="понял, утренняя"),
        HumanMessage(content="а ещё бывает вечерняя тяга"),
        AIMessage(content="понял оба варианта"),
    ]
    live = "нет, я имел в виду вечернюю, исправляю утреннее предположение"
    context = build_conversation_context(history, "", live)
    assert [item.role for item in context.messages] == ["user", "assistant", "user", "assistant"]
    assert [item.provenance for item in context.messages] == [
        "human",
        "assistant",
        "human",
        "assistant",
    ]
    assert "вечерняя" in context.messages[2].text
    assert context.user_message == live


def test_answer_messages_carry_resolved_intent_verbatim_user_message() -> None:
    from aa.conversation.prompt_builder import build_answer_messages

    live = "вернёмся к alpha-topic-marker?"
    intent = "standalone intent: вернуться к утренней теме alpha-topic-marker"
    messages = build_answer_messages(
        recent=[HumanMessage(content="старое")],
        summary="резюме",
        passages=[],
        user_message=live,
        resolved_intent=intent,
    )
    final = str(messages[-1].content)
    assert "<resolved_intent>" in final
    assert intent in final
    assert f"<user_message>\n{live}\n</user_message>" in final


def test_information_needs_are_typed_with_stable_ids() -> None:
    from aa.conversation.conversation_context import (
        build_conversation_context,
        build_resolved_turn,
        needs_from_plan,
    )

    needs = needs_from_plan("standalone intent", ["запрос один", "запрос два"])
    assert [item.need_id for item in needs] == ["need-1", "need-2", "need-3"]
    assert all(item.text.strip() for item in needs)
    # Stable through replay: same plan order yields same ids.
    again = needs_from_plan("standalone intent", ["запрос один", "запрос два"])
    assert [item.need_id for item in again] == [item.need_id for item in needs]
    # Conversational plans carry no needs and never claim fulfillment.
    assert needs_from_plan("", []) == []

    context = build_conversation_context([], "", "привет")
    turn = build_resolved_turn(
        user_message="привет",
        resolved_intent="standalone intent",
        context=context,
        information_needs=needs,
        planner_mode="retrieval",
        search_queries=["запрос один", "запрос два"],
    )
    dumped = turn.model_dump(mode="json")
    assert dumped["user_message"] == "привет"
    assert dumped["resolved_intent"] == "standalone intent"
    assert dumped["context_digest"]
    assert [item["need_id"] for item in dumped["information_needs"]] == [
        "need-1",
        "need-2",
        "need-3",
    ]


def test_all_stages_prove_same_digest_and_mismatch_fails_closed() -> None:
    from langchain_core.messages import HumanMessage as _HM

    from aa.conversation.conversation_context import (
        assert_same_context_digest,
        build_conversation_context,
        build_resolved_turn,
        canonical_model_view,
        context_digest_for_canonical,
        needs_from_plan,
    )
    from aa.conversation.finalization import (
        FinalizationError,
        candidate_from_state,
    )
    from aa.conversation.retrieval_node import (
        _selection_context_from_state,
        selection_context_digest,
    )

    question = "что помогает при тяге вечером?"
    history = [_HM(content="раньше говорили про утро")]
    context = build_conversation_context(history, "", question)
    needs = needs_from_plan(question, [question])
    turn = build_resolved_turn(
        user_message=question,
        resolved_intent=question,
        context=context,
        information_needs=needs,
        planner_mode="retrieval",
        search_queries=[question],
    )
    state: dict[str, Any] = {
        "messages": [_HM(content="раньше говорили про утро"), _HM(content=question)],
        "current_user_message": question,
        "conversation_summary": "",
        "resolved_intent": question,
        "conversation_context": context.model_dump(mode="json"),
        "resolved_turn": turn.model_dump(mode="json"),
        "information_needs": [item.model_dump(mode="json") for item in needs],
        "context_digest": turn.context_digest,
    }
    assert selection_context_digest(state) == turn.context_digest  # type: ignore
    assert assert_same_context_digest(state, stage="selector") == turn.context_digest
    intent, combined, live = _selection_context_from_state(state)  # type: ignore
    assert intent == question
    assert live == question
    assert combined == canonical_model_view(context)["combined"]

    # Finalizer binds the same canonical digest.
    state.update(
        {
            "evidence_pack": [],
            "final_response": "Добрый вечер! Поддержка рядом помогает спокойно.",
            "draft_response": "Добрый вечер! Поддержка рядом помогает спокойно.",
            "grounding_result": {"verified": False, "units": [], "all_required_supported": False},
            "retry_state": {},
            "route": "normal",
        }
    )
    candidate, _g, _q, _i, _s, messages, _t = candidate_from_state(state)
    assert candidate.context_digest == turn.context_digest
    assert candidate.context_digest == context_digest_for_canonical(context)

    # Divergent stage snapshot fails closed instead of certifying.
    tampered = dict(state)
    tampered["context_digest"] = "0" * 64
    with pytest.raises(FinalizationError):
        candidate_from_state(tampered)


def test_summary_never_becomes_book_evidence() -> None:
    import hashlib

    import pytest

    from aa.conversation.evidence_integrity import (
        EvidencePackIntegrityError,
        validate_book_pack_for_model_use,
    )
    from aa.conversation.finalization import AnswerCandidate, FinalizationError, certify_candidate

    summary_claim = "В книге сказано: луна лечит тягу за вечер."
    # A summary-shaped dict without canonical passage provenance is not an
    # evidence bundle and fails integrity instead of grounding a claim.
    with pytest.raises((EvidencePackIntegrityError, Exception)):
        validate_book_pack_for_model_use(
            [{"passage_id": "", "text": summary_claim, "source_id": "", "section_id": ""}]
        )

    text = "Поддержка рядом помогает пережить тягу спокойно."
    digest = "0" * 64
    candidate = AnswerCandidate(
        text=text, evidence_bundle=[], context_digest=digest, outcome_kind="answer"
    )
    with pytest.raises(FinalizationError):
        certify_candidate(
            candidate=candidate,
            grounding_result={"all_required_supported": True, "units": []},
            question="что помогает?",
            resolved_intent="что помогает?",
            summary=summary_claim,
            recent_messages=[],
        )
    _ = hashlib.sha256(b"x").hexdigest()


def test_proactive_summary_triggers_before_window_falls_out() -> None:
    from langchain_core.messages import HumanMessage as _HM

    from aa.conversation.conversation_context import needs_proactive_summary

    short = [_HM(content="привет")]
    assert needs_proactive_summary(short, summary="") is False
    many = [_HM(content=f"сообщение {index}") for index in range(30)]
    assert needs_proactive_summary(many, summary="") is True


async def test_graph_builds_context_once_before_planner() -> None:
    from langchain_core.messages import HumanMessage as _HM
    from langchain_core.runnables import RunnableLambda

    from aa.conversation.graph import build_turn_graph

    seen: list[Any] = []

    async def _plan(messages: Any) -> Any:
        seen.append(messages)
        return {"mode": "retrieval", "resolved_intent": "живой запрос", "queries": ["запрос один"]}

    graph = build_turn_graph(planner_model=RunnableLambda(_plan))
    history = _history_with_old_topic()
    state = {
        "messages": history + [_HM(content="живой запрос")],
        "current_user_message": "живой запрос",
        "conversation_summary": "",
    }
    result = await graph.ainvoke(state)  # type: ignore
    assert result["planner_invoked"] is True
    assert result["conversation_context"]["snapshot_version"] == "canonical-conversation-context/1"
    assert result["resolved_turn"]["user_message"] == "живой запрос"
    assert result["resolved_turn"]["resolved_intent"] == "живой запрос"
    assert result["context_digest"] == result["resolved_turn"]["context_digest"]
    assert [item["need_id"] for item in result["information_needs"]][0] == "need-1"
    assert len(seen) == 1


import pytest  # noqa: E402  (kept last so test collection order stays stable)
