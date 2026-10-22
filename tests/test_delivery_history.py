"""Delivery confirmation governs the history seen by the next dialogue turn."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from aa.conversation.delivery_history import (
    DELIVERY_METADATA,
    delivery_history_updates,
    provisional_answer,
)
from aa.conversation.finalization import build_delivery_receipts
from aa.conversation.graph_runtime import GraphTurnRuntime
from aa.conversation.graph_state import TurnState


def _receipts(text: str, *, segments: list[str] | None = None) -> list[dict[str, Any]]:
    return [
        receipt.model_dump(mode="json")
        for receipt in build_delivery_receipts(
            certified_text=text,
            segments=segments or [text],
            certificate_id="cert",
            turn_id="thread",
            channel="sendMessage",
            retry_id="attempt",
            status="confirmed",
        )
    ]


@pytest.mark.parametrize("status", ["failed", "unknown"])
def test_unconfirmed_answer_is_removed_without_touching_other_history(status: str) -> None:
    answer = provisional_answer("Фиктивный ответ.", "cert")
    receipts = _receipts(str(answer.content))
    receipts[0]["status"] = status
    history: list[BaseMessage] = [HumanMessage(content="Вопрос."), AIMessage(content="Ранее."), answer]
    updates = delivery_history_updates(history, receipts, thread="thread", drop_unconfirmed=True)
    assert len(updates) == 1
    assert isinstance(updates[0], RemoveMessage)
    assert updates[0].id == answer.id


def test_partial_delivery_cannot_confirm_full_answer() -> None:
    text = "Первая часть. Вторая часть."
    receipts = _receipts(text, segments=["Первая часть.", "Вторая часть."])
    receipts[1]["status"] = "failed"
    answer = provisional_answer(text, "cert")
    committed = delivery_history_updates(
        [answer], receipts, thread="thread", drop_unconfirmed=False
    )
    assert committed == []
    updates = delivery_history_updates([answer], receipts, thread="thread", drop_unconfirmed=True)
    assert isinstance(updates[0], RemoveMessage)


@pytest.mark.parametrize("field", ["turn_id", "certificate_id", "final_sha256"])
def test_receipt_from_another_answer_cannot_promote_history(field: str) -> None:
    answer = provisional_answer("Фиктивный ответ.", "cert")
    receipts = _receipts(str(answer.content))
    receipts[0][field] = "different"
    updates = delivery_history_updates([answer], receipts, thread="thread", drop_unconfirmed=True)
    assert isinstance(updates[0], RemoveMessage)


@pytest.mark.parametrize("delivered", [False, True])
async def test_next_graph_turn_only_sees_confirmed_answers(delivered: bool) -> None:
    seen: list[list[BaseMessage]] = []

    async def capture(state: TurnState) -> dict[str, Any]:
        seen.append(list(state["messages"]))
        return {}

    builder = StateGraph(TurnState)
    builder.add_node("capture", capture)
    builder.add_edge(START, "capture")
    builder.add_edge("capture", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "thread"}}
    answer = provisional_answer("Фиктивный ответ без цитат.", "cert")
    await graph.ainvoke({"messages": [HumanMessage(content="Вопрос."), answer]}, config)
    runtime = GraphTurnRuntime(graph=graph)
    if delivered:
        runtime.record_delivery_receipts("thread", _receipts(str(answer.content)))
        # No quote ranges exist: confirmation must still persist the answer.
        assert await runtime.apply_delivery_commit("thread") == []
        # Receipt confirmation survives recreation of the runtime wrapper.
        runtime = GraphTurnRuntime(graph=graph)
    await runtime._invoke_graph(graph, "thread", "Следующий вопрос.")
    replies = [message for message in seen[-1] if isinstance(message, AIMessage)]
    assert len(replies) == int(delivered)
    if delivered:
        assert replies[0].additional_kwargs[DELIVERY_METADATA]["status"] == "confirmed"
        assert replies[0].content == answer.content
    assert seen[-1][-1].content == "Следующий вопрос."
