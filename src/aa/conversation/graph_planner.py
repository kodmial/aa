"""Mandatory hidden query planner for the new turn graph (issue #113).

The planner runs on every normal conversational turn after deterministic
application-command/safety handling. Its output schema is exactly
``QueryPlan { queries: list[str] }`` with valid lengths 0 or 10..16 distinct
Russian queries. There are no ``intent``, ``category``, ``is_substantive``,
``confidence``, or taxonomy fields.

Structured decoding uses LangChain's standard ``PydanticOutputParser`` plus
the framework retry facility (``Runnable.with_retry``). No ad-hoc JSON
parser or retry loop is implemented here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, field_validator

from aa.conversation.graph_state import TurnState

logger = logging.getLogger("aa.conversation.graph_planner")

QUERY_PLANNER_VERSION = "query-planner-system-v1"
MIN_NONEMPTY_QUERIES = 10
MAX_QUERIES = 16


def _prompts_dir() -> Path:
    """Return the repository ``prompts/`` directory."""
    return Path(__file__).resolve().parents[3] / "prompts"


def load_query_planner_prompt() -> str:
    """Load the versioned English planner system prompt (#112 verbatim)."""
    return (_prompts_dir() / f"{QUERY_PLANNER_VERSION}.md").read_text(encoding="utf-8").strip()


class QueryPlan(BaseModel):
    """Minimal structured planner output owned by AA."""

    model_config = ConfigDict(extra="forbid")

    queries: list[str]

    @field_validator("queries")
    @classmethod
    def _check_queries(cls, value: list[Any]) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("queries must be a list of strings")
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in value:
            if not isinstance(raw, str):
                raise ValueError("each query must be a string")
            normalized = " ".join(raw.strip().split())
            if not normalized:
                raise ValueError("queries must not contain empty entries")
            key = normalized.casefold()
            if key in seen:
                raise ValueError("queries must not contain exact duplicates")
            seen.add(key)
            cleaned.append(normalized)
        if len(cleaned) == 0:
            return []
        if not MIN_NONEMPTY_QUERIES <= len(cleaned) <= MAX_QUERIES:
            raise ValueError("non-empty plans require 10..16 distinct queries")
        return cleaned


def validate_query_plan(data: dict[str, Any]) -> QueryPlan:
    """Validate a raw planner mapping against the minimal schema."""
    return QueryPlan.model_validate(data)


def format_planner_context(state: TurnState, *, max_messages: int = 20) -> str:
    """Render managed conversation context for the planner (data only)."""
    summary = str(state.get("conversation_summary", "") or "")
    messages: list[BaseMessage] = list(state.get("messages", []) or [])
    recent = messages[-max_messages:] if max_messages > 0 else messages
    parts: list[str] = []
    if summary.strip():
        parts.append(f"Conversation summary:\n{summary.strip()}")
    if recent:
        parts.append(f"Recent conversation:\n{get_buffer_string(recent)}")
    return "\n\n".join(parts).strip()


def build_planner_chain(model: BaseChatModel) -> Runnable[dict[str, Any], QueryPlan]:
    """Build the planner chain with standard parser + retry facilities."""
    parser = PydanticOutputParser(pydantic_object=QueryPlan)
    system_prompt = load_query_planner_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\n{format_instructions}"),
            (
                "human",
                "Conversation context:\n{planner_context}\n\n"
                "Current user message:\n{current_message}",
            ),
        ]
    )
    chain: Runnable[dict[str, Any], QueryPlan] = (
        prompt.partial(format_instructions=parser.get_format_instructions()) | model | parser
    )
    return chain.with_retry(stop_after_attempt=3)


async def run_planner_chain(
    state: TurnState,
    *,
    model: BaseChatModel,
) -> QueryPlan:
    """Invoke the planner chain for ``state`` (hidden orchestration call)."""
    chain = build_planner_chain(model)
    planner_context = format_planner_context(state)
    current_message = str(state.get("current_user_message", "") or "")
    if not current_message.strip():
        raise ValueError("current_user_message must not be empty")
    # Never log the message or the context; only the fact of invocation.
    logger.info("planner invocation started")
    plan = await chain.ainvoke(
        {"planner_context": planner_context, "current_message": current_message}
    )
    logger.info("planner invocation completed", extra={"query_count": len(plan.queries)})
    return plan


async def planner_node(
    state: TurnState,
    *,
    model: BaseChatModel,
) -> dict[str, Any]:
    """LangGraph planner node: hidden state update, never a chat message."""
    plan = await run_planner_chain(state, model=model)
    return {"search_queries": list(plan.queries)}


__all__ = [
    "MAX_QUERIES",
    "MIN_NONEMPTY_QUERIES",
    "QUERY_PLANNER_VERSION",
    "QueryPlan",
    "build_planner_chain",
    "format_planner_context",
    "load_query_planner_prompt",
    "planner_node",
    "run_planner_chain",
    "validate_query_plan",
]
