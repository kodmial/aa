"""P0 kodmial/aa#278: Gate C live-production-path continuation failure.

Proven product failure on exact main a0f4214 (run 37846898690):

- C:live-continuation-helpful-followup-ellipsis-4 (plus the held-out
  sibling -13, live-context-switch-helpful and the resulting
  live-answer-no-generic-collapse with 5 retry fallbacks and 4
  adequacy-repair-failed turns while 8/8 substantive first turns
  grounded fine).

Root cause: the LangGraph answer node never wrote the served assistant
reply back into thread ``messages``. The checkpointer therefore kept
user turns only, so the next turn's planner/answer/verifier resolved
terse follow-ups and ellipsis against user text alone. A follow-up
whose antecedent lives in the prior assistant answer ("why is this
important?", "what should I do with this?") resolved to a generic
intent, retrieved a drifting pack, failed whole-turn adequacy and
collapsed to the retry fallback. Topic-carrying turns passed, which is
exactly the observed split.

Repair: persist every served assistant reply as an ``AIMessage`` from
``answer_pipeline_node`` so thread memory holds full user+assistant
history. Generic, no exact-question special cases; Product Contract
#110 unchanged.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda


class _GlueAnswer:
    """Deterministic conversational-glue answer model."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        return AIMessage(content=self._text)


class _GlueVerifier:
    """Verifier fake: pure glue needs no book evidence."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return {
            "requires_book_evidence": False,
            "supported": True,
            "evidence_passage_ids": [],
            "addresses_intent": True,
        }


def _conversational_plan() -> dict[str, Any]:
    return {"mode": "conversational", "resolved_intent": "", "queries": []}


def _script_planner(seen: list[list[BaseMessage]]) -> Any:
    queue = [_conversational_plan(), _conversational_plan()]

    def _reply(messages: list[BaseMessage]) -> Any:
        seen.append(list(messages))
        return queue.pop(0)

    return RunnableLambda(_reply)


async def test_answer_pipeline_node_persists_assistant_reply() -> None:
    import hashlib

    from aa.conversation.turn_pipeline import answer_pipeline_node

    reply_text = "Поддержка рядом помогает пережить тягу сегодня."
    pack_text = reply_text
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

    class _BookAnswer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content=reply_text)

    class _BookVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": True,
            }

    queries = [f"запрос про поддержку {index}" for index in range(12)]
    result = await answer_pipeline_node(
        {
            "route": "normal",
            "current_user_message": "frozen-278-marker-alpha turn",
            "messages": [HumanMessage(content="frozen-278-marker-alpha turn")],
            "search_queries": queries,
            "evidence_pack": pack,
            "conversation_summary": "",
            "planner_mode": "retrieval",
            "resolved_intent": "frozen-278-marker-alpha intent",
            "retry_state": {
                "planner_reason": "substantive-with-queries",
                "planner_outcome": "ok",
                "planner_mode": "retrieval",
                "resolved_intent": "frozen-278-marker-alpha intent",
            },
        },
        answer_model=_BookAnswer(),
        verifier_model=_BookVerifier(),
        planner_model=None,
        retrieval_index=None,
    )
    assert result["final_response"] == reply_text
    stored = result["messages"]
    assert len(stored) == 1
    assert isinstance(stored[0], AIMessage)
    assert str(stored[0].content) == reply_text


async def test_followup_planner_sees_prior_assistant_answer(tmp_path: Any) -> None:
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig

    from aa.conversation.graph import build_turn_graph, turn_input
    from aa.conversation.memory import (
        MemoryConfig,
        SqliteCheckpointerFactory,
        thread_id_for_chat,
    )

    _ = Path

    seen: list[list[BaseMessage]] = []
    first_reply = "Понял вас. Поддержка рядом помогает. Что сейчас важнее?"
    second_reply = "Продолжаем спокойно. Что сейчас важнее всего?"

    answers = [first_reply, second_reply]

    class _TwoAnswers:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content=answers.pop(0))

    factory = SqliteCheckpointerFactory(MemoryConfig(checkpoint_dir=tmp_path))
    async with factory.checkpointer() as saver:
        graph = build_turn_graph(
            planner_model=_script_planner(seen),
            checkpointer=saver,
            answer_model=_TwoAnswers(),
            verifier_model=_GlueVerifier(),
        )
        config: RunnableConfig = {"configurable": {"thread_id": thread_id_for_chat(771278)}}
        first = await graph.ainvoke(turn_input("первое сообщение про вечер"), config=config)
        assert first["final_response"] == first_reply
        roles_first = [item.type for item in first["messages"]]
        assert "ai" in roles_first

        second = await graph.ainvoke(turn_input("а почему это важно?"), config=config)
        assert second["final_response"] == second_reply
        # The follow-up planner input must carry the prior assistant
        # answer, not user turns alone; otherwise ellipsis ("это")
        # cannot resolve to its antecedent.
        assert len(seen) == 2
        followup_prompt = "\n".join(str(item.content) for item in seen[1])
        assert "первое сообщение про вечер" in followup_prompt
        assert first_reply in followup_prompt
        # Thread state holds the full alternating history.
        types = [item.type for item in second["messages"]]
        assert types.count("human") == 2
        assert types.count("ai") == 2


def test_no_exact_question_special_cases() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
    ]
    for source in sources:
        for fragment in (
            "почему это вообще важно",
            "что мне делать-то",
            "ночью не могу уснуть",
            "покупать акции",
        ):
            assert fragment not in source
