"""Managed conversation memory for the new turn graph (issue #113).

Token-driven compaction only: compaction starts at roughly 60% of the
~200k context reference and retains roughly the newest 20% verbatim, with
headroom for book evidence, verification, and final generation. The
framework's ``SummarizationMiddleware`` performs compaction with the
AA-specific English summarization prompt; this module only owns thresholds,
deterministic thread identity, and explicit checkpointer wiring.

One Telegram private chat maps deterministically to one graph thread
identity. The raw chat identifier is never logged.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

logger = logging.getLogger("aa.conversation.graph_memory")

CONTEXT_REFERENCE_TOKENS = 200_000
COMPACTION_TRIGGER_TOKENS = 120_000
RETAIN_TOKENS = 40_000

SUMMARY_PROMPT_VERSION = "conversation-summary-v1"
THREAD_NAMESPACE = "tg-private-chat-v1"


def _prompts_dir() -> Path:
    """Return the repository ``prompts/`` directory."""
    return Path(__file__).resolve().parents[3] / "prompts"


def load_summarization_prompt() -> str:
    """Load the versioned English summarization prompt (#112 verbatim)."""
    return (_prompts_dir() / f"{SUMMARY_PROMPT_VERSION}.md").read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class MemoryConfig:
    """Token-driven compaction thresholds (eval-tunable, not constants)."""

    context_reference_tokens: int = CONTEXT_REFERENCE_TOKENS
    trigger_tokens: int = COMPACTION_TRIGGER_TOKENS
    retain_tokens: int = RETAIN_TOKENS


DEFAULT_MEMORY_CONFIG = MemoryConfig()


@dataclass(frozen=True)
class CheckpointerConfig:
    """Isolated storage backend selection for the graph checkpointer."""

    kind: str = "memory"


def create_checkpointer(config: CheckpointerConfig | None = None) -> Any:
    """Create the framework-supported local checkpointer (testable).

    The ``memory`` backend uses LangGraph's ``MemorySaver`` for the current
    single-worker runtime. Production storage can be changed later by adding
    a new ``kind`` here without changing graph semantics.
    """
    selected = config or CheckpointerConfig()
    if selected.kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    raise ValueError(f"unsupported checkpointer backend: {selected.kind!r}")


def thread_id_for_chat(chat_id: int) -> str:
    """Map one Telegram private chat to one stable graph thread identity."""
    digest = hashlib.sha256(f"{THREAD_NAMESPACE}:{chat_id}".encode()).hexdigest()
    return f"tg-{digest[:32]}"


def _message_text(item: Any) -> str:
    """Extract countable text from one conversation input item."""
    if isinstance(item, BaseMessage):
        content = item.content
        if isinstance(content, str):
            return content
        return str(content)
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return f"{item[0]}: {item[1]}"
    if isinstance(item, dict):
        return str(item.get("content", str(item)))
    return str(item)


def count_conversation_tokens(items: Iterable[Any]) -> int:
    """Count tokens for conversation items (conservative ceiling)."""
    from aa.corpus.budget import estimate_text_tokens

    total = 0
    for item in items:
        total += estimate_text_tokens(_message_text(item))
    return total


def needs_compaction(token_count: int, *, config: MemoryConfig | None = None) -> bool:
    """Return whether token usage has reached the compaction trigger."""
    selected = config or DEFAULT_MEMORY_CONFIG
    return token_count >= selected.trigger_tokens


def split_for_compaction(
    messages: list[BaseMessage],
    *,
    config: MemoryConfig | None = None,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split ``messages`` into ``(to_summarize, to_keep)`` by token budget.

    The newest ``retain_tokens`` stay verbatim. Structured pairs are never
    split: when the keep window would orphan an ``AIMessage`` with tool calls
    or start with a tool result, the boundary moves to keep the whole unit
    together.
    """
    from aa.corpus.budget import estimate_text_tokens

    selected = config or DEFAULT_MEMORY_CONFIG
    if not messages:
        return [], []
    keep: list[BaseMessage] = []
    used = 0
    for message in reversed(messages):
        content = message.content if isinstance(message.content, str) else str(message.content)
        need = estimate_text_tokens(content)
        if keep and used + need > selected.retain_tokens:
            break
        keep.append(message)
        used += need
    keep.reverse()
    boundary = len(messages) - len(keep)
    # Never orphan tool-call/result pairs at the boundary.
    while 0 < boundary < len(messages):
        first_kept = keep[0] if keep else None
        previous = messages[boundary - 1]
        if isinstance(first_kept, ToolMessage):
            keep.insert(0, previous)
            boundary -= 1
            continue
        if isinstance(previous, AIMessage) and getattr(previous, "tool_calls", None):
            keep.insert(0, previous)
            boundary -= 1
            continue
        if isinstance(first_kept, AIMessage) and isinstance(previous, HumanMessage):
            # Keep a leading AI reply together with its human turn only when
            # the AI message alone would otherwise lose its anchor. The
            # default keeps the pair together when the human turn is small.
            human_text = (
                previous.content if isinstance(previous.content, str) else str(previous.content)
            )
            if estimate_text_tokens(human_text) + used <= selected.retain_tokens * 2:
                keep.insert(0, previous)
                boundary -= 1
                continue
        break
    return messages[:boundary], keep


def build_summarization_middleware(
    model: BaseChatModel, *, config: MemoryConfig | None = None
) -> Any:
    """Build the framework summarization middleware with AA configuration."""
    from langchain.agents.middleware.summarization import SummarizationMiddleware

    selected = config or DEFAULT_MEMORY_CONFIG
    middleware = SummarizationMiddleware(
        model,
        trigger=("tokens", selected.trigger_tokens),
        keep=("tokens", selected.retain_tokens),
        token_counter=count_conversation_tokens,
        summary_prompt=load_summarization_prompt(),
    )
    logger.info("summarization middleware configured")
    return middleware


def build_summarizer_chain(model: BaseChatModel) -> Any:
    """Build the hidden summarizer chain with the AA continuity prompt."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", load_summarization_prompt()), ("human", "{conversation_text}")]
    )
    chain = prompt | model | StrOutputParser()
    return chain.with_retry(stop_after_attempt=3)


async def run_summarization(
    messages_to_summarize: list[BaseMessage],
    *,
    model: BaseChatModel,
) -> str:
    """Summarize older messages for continuity (hidden call, no history write)."""
    from langchain_core.messages import get_buffer_string

    if not messages_to_summarize:
        return ""
    chain = build_summarizer_chain(model)
    # Never log message content; only the fact of invocation.
    logger.info("summarizer invocation started")
    summary = await chain.ainvoke({"conversation_text": get_buffer_string(messages_to_summarize)})
    logger.info("summarizer invocation completed")
    return str(summary).strip()


__all__ = [
    "COMPACTION_TRIGGER_TOKENS",
    "CONTEXT_REFERENCE_TOKENS",
    "DEFAULT_MEMORY_CONFIG",
    "RETAIN_TOKENS",
    "SUMMARY_PROMPT_VERSION",
    "CheckpointerConfig",
    "MemoryConfig",
    "build_summarization_middleware",
    "build_summarizer_chain",
    "count_conversation_tokens",
    "create_checkpointer",
    "load_summarization_prompt",
    "needs_compaction",
    "run_summarization",
    "split_for_compaction",
    "thread_id_for_chat",
]
