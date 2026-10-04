"""Managed conversation memory for the v2 turn graph (issue #113).

Framework-managed state, not a custom memory database:

- a LangGraph checkpointer owns durable thread/session state; one Telegram
  private chat maps deterministically to one graph thread identity via a
  one-way digest (the raw chat identifier is never logged or stored);
- token-based compaction uses the LangChain summarization middleware's
  trigger/keep facilities with the #112 baselines (compact near ~60% of
  the ~200k reference, retain ~20% verbatim); decisions are token-driven,
  never fixed-turn-count driven;
- the checkpointer backend sits behind a replaceable factory so
  production storage can change later without changing graph semantics.

The checkpoint database lives only in job-private runtime storage and is
removed on normal bounded-runtime shutdown; it is never uploaded, cached
or shipped as an artifact. No prompt, user text, summary or identifier is
logged here: only token counts and compaction decisions.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from aa.conversation.v2_prompts import load_summarizer_system_v2

logger = logging.getLogger("aa.conversation.memory")

CONTEXT_REFERENCE_TOKENS = 200_000
COMPACTION_TRIGGER_TOKENS = 120_000
RECENT_KEEP_TOKENS = 40_000

CHECKPOINT_ENV_VAR = "AA_V2_CHECKPOINT_DIR"
CHECKPOINT_FILENAME = "aa-v2-checkpoints.sqlite3"


def default_checkpoint_dir() -> Path:
    """Resolve job-private runtime storage for the checkpoint database."""
    override = os.environ.get(CHECKPOINT_ENV_VAR, "").strip()
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / f"aa-v2-checkpoints-{os.getpid()}"


def thread_id_for_chat(chat_id: int) -> str:
    """Map one Telegram private chat deterministically to one thread id.

    A one-way digest keeps the raw chat identifier out of stored state and
    out of logs; the mapping itself is never logged.
    """
    digest = hashlib.sha256(f"telegram-private:{chat_id}".encode()).hexdigest()[:32]
    return f"aa-v2-{digest}"


@dataclass(frozen=True)
class MemoryConfig:
    """Starting defaults for token-driven compaction (eval-tunable)."""

    checkpoint_dir: Path
    context_reference_tokens: int = CONTEXT_REFERENCE_TOKENS
    trigger_tokens: int = COMPACTION_TRIGGER_TOKENS
    keep_tokens: int = RECENT_KEEP_TOKENS
    checkpoint_filename: str = CHECKPOINT_FILENAME


def default_memory_config(*, checkpoint_dir: Path | None = None) -> MemoryConfig:
    """Build the production starting memory configuration."""
    return MemoryConfig(checkpoint_dir=checkpoint_dir or default_checkpoint_dir())


class CheckpointerFactory(Protocol):
    """Replaceable storage backend boundary for graph checkpoints."""

    @asynccontextmanager
    async def checkpointer(self) -> AsyncIterator[AsyncSqliteSaver]:
        """Yield a ready checkpointer; cleanup happens on context exit."""
        raise NotImplementedError
        yield  # pragma: no cover - protocol stub


class SqliteCheckpointerFactory:
    """Local single-worker SQLite backend for the current runtime."""

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config

    @property
    def config(self) -> MemoryConfig:
        """The memory configuration owned by this factory."""
        return self._config

    @property
    def db_path(self) -> Path:
        """Filesystem location of the conversation database (private)."""
        return self._config.checkpoint_dir / self._config.checkpoint_filename

    @asynccontextmanager
    async def checkpointer(self) -> AsyncIterator[AsyncSqliteSaver]:
        """Yield an ``AsyncSqliteSaver`` bound to the private database file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(self.db_path)) as saver:
            yield saver

    def cleanup(self) -> None:
        """Remove job-private checkpoint storage on bounded-runtime shutdown."""
        shutil.rmtree(self._config.checkpoint_dir, ignore_errors=True)
        logger.info("v2 checkpoint storage cleaned")


def build_summarization_middleware(
    model: BaseChatModel | str,
    *,
    config: MemoryConfig | None = None,
) -> SummarizationMiddleware:
    """Configure the framework middleware with the #112 AA memory contract.

    Token-driven trigger/keep come from the framework's own facilities;
    the summarization prompt is the versioned English #112 artifact so
    compacted memory preserves continuity without becoming AA doctrine.
    """
    resolved = config or default_memory_config()
    return SummarizationMiddleware(
        model=model,
        trigger=("tokens", resolved.trigger_tokens),
        keep=("tokens", resolved.keep_tokens),
        token_counter=count_tokens_approximately,
        summary_prompt=load_summarizer_system_v2(),
    )


def count_message_tokens(messages: list[BaseMessage]) -> int:
    """Count tokens with the framework counter (no custom bookkeeping)."""
    return int(count_tokens_approximately(messages))


def needs_compaction(messages: list[BaseMessage], *, trigger_tokens: int) -> bool:
    """Token-driven compaction decision; never fixed-turn-count driven."""
    return count_message_tokens(messages) >= trigger_tokens


def split_keep_window(messages: list[BaseMessage], *, keep_tokens: int) -> int:
    """Return the split index keeping roughly the newest ``keep_tokens``.

    The cut never orphans related structured messages: when the oldest
    retained message answers a tool call, the calling assistant message is
    retained with it.
    """
    if not messages:
        return 0
    total = 0
    start = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        total += count_message_tokens([messages[index]])
        if total > keep_tokens and index > 0:
            start = index
            break
        start = index
    else:
        return 0
    while start > 0:
        oldest = messages[start]
        previous = messages[start - 1]
        if oldest.type == "tool" and isinstance(previous, AIMessage) and previous.tool_calls:
            start -= 1
            continue
        break
    return start


def render_messages_for_summary(messages: list[BaseMessage]) -> str:
    """Render a window of real dialogue for the summarizer model call."""
    lines: list[str] = []
    for message in messages:
        role = "user" if message.type == "human" else "assistant"
        content = message.content
        text = content if isinstance(content, str) else str(content)
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


async def summarize_window(
    messages: list[BaseMessage],
    *,
    model: Runnable[list[BaseMessage], BaseMessage],
    summary_prompt: str | None = None,
) -> str:
    """Summarize one dialogue window for continuity (hidden model call)."""
    prompt = summary_prompt or load_summarizer_system_v2()
    rendered = render_messages_for_summary(messages)
    if not rendered.strip():
        return ""
    reply = await model.ainvoke([SystemMessage(content=prompt), HumanMessage(content=rendered)])
    content = reply.content if isinstance(reply, BaseMessage) else getattr(reply, "content", "")
    return content.strip() if isinstance(content, str) else str(content).strip()


async def maybe_compact_state(
    *,
    messages: list[BaseMessage],
    previous_summary: str,
    model: Runnable[list[BaseMessage], BaseMessage],
    config: MemoryConfig,
) -> tuple[str, list[BaseMessage]]:
    """Compact older dialogue when the token trigger is reached.

    Returns ``(summary, retained_messages)``. Below the trigger the inputs
    pass through unchanged. The summary preserves conversational
    referents/continuity; it must never be treated as AA evidence (the
    prompt contract and the answer node enforce that downstream).
    """
    if not needs_compaction(messages, trigger_tokens=config.trigger_tokens):
        return previous_summary, messages
    split = split_keep_window(messages, keep_tokens=config.keep_tokens)
    older = messages[:split] if split > 0 else messages[:-1] if len(messages) > 1 else messages
    retained = messages[split:] if split > 0 else messages[-1:]
    window_summary = await summarize_window(older, model=model)
    if not window_summary:
        logger.info("v2 compaction skipped", extra={"reason": "empty-summary"})
        return previous_summary, retained
    if previous_summary.strip():
        merged = f"{previous_summary.strip()}\n{window_summary}"
    else:
        merged = window_summary
    logger.info(
        "v2 conversation compacted",
        extra={
            "older_messages": len(older),
            "retained_messages": len(retained),
            "summary_tokens": count_message_tokens([HumanMessage(content=merged)]),
        },
    )
    return merged, retained


__all__ = [
    "CHECKPOINT_ENV_VAR",
    "CHECKPOINT_FILENAME",
    "COMPACTION_TRIGGER_TOKENS",
    "CONTEXT_REFERENCE_TOKENS",
    "RECENT_KEEP_TOKENS",
    "CheckpointerFactory",
    "MemoryConfig",
    "SqliteCheckpointerFactory",
    "build_summarization_middleware",
    "count_message_tokens",
    "default_checkpoint_dir",
    "default_memory_config",
    "maybe_compact_state",
    "needs_compaction",
    "render_messages_for_summary",
    "split_keep_window",
    "summarize_window",
    "thread_id_for_chat",
]
