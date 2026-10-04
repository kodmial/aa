"""Managed conversation memory for the v2 turn graph (issue #113).

Framework-managed state, not a custom memory database:

- a LangGraph checkpointer owns durable thread/session state; one Telegram
  private chat maps deterministically to one graph thread identity via a
  one-way digest (the raw chat identifier is never logged or stored);
- token-based compaction uses the LangMem graph-native ``SummarizationNode``
  / ``RunningSummary`` mechanism with the #112 baselines (compact near ~60%
  of the ~200k reference, retain ~20% verbatim model-input view, 4k summary
  ceiling); decisions are token-driven, never fixed-turn-count driven;
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
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langmem.short_term import RunningSummary, SummarizationNode  # type: ignore[import-untyped]

from aa.conversation.v2_prompts import load_summarizer_system_v2

logger = logging.getLogger("aa.conversation.memory")

CONTEXT_REFERENCE_TOKENS = 200_000
COMPACTION_TRIGGER_TOKENS = 120_000
RECENT_KEEP_TOKENS = 40_000
MAX_SUMMARY_TOKENS = 4_096

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
    max_summary_tokens: int = MAX_SUMMARY_TOKENS
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


def _summarizer_prompts() -> tuple[ChatPromptTemplate, ChatPromptTemplate]:
    """Build LangMem prompts around the versioned English #112 artifact."""
    system_text = load_summarizer_system_v2()
    initial = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            MessagesPlaceholder("messages"),
            ("human", "Create a compact continuity summary of the conversation above."),
        ]
    )
    existing = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            MessagesPlaceholder("messages"),
            (
                "human",
                "This is the running summary so far: {existing_summary}\n"
                "Extend it with the new messages above, preserving continuity "
                "without adding AA doctrine or advice.",
            ),
        ]
    )
    return initial, existing


def build_summarization_node(
    model: Any,
    *,
    config: MemoryConfig | None = None,
) -> SummarizationNode:
    """Configure the LangMem graph-native summarization node.

    Token budgets come from the #112 AA memory contract: trigger near ~60%
    of the reference, recent/raw + summary envelope ~20%, summary ceiling
    4k. The summarization prompt is the versioned English #112 artifact so
    compacted memory preserves continuity without becoming AA doctrine.
    """
    resolved = config or default_memory_config()
    initial, existing = _summarizer_prompts()
    return SummarizationNode(
        model=model,
        max_tokens=resolved.keep_tokens,
        max_tokens_before_summary=resolved.trigger_tokens,
        max_summary_tokens=resolved.max_summary_tokens,
        token_counter=count_tokens_approximately,
        initial_summary_prompt=initial,
        existing_summary_prompt=existing,
        input_messages_key="messages",
        output_messages_key="messages",
    )


def count_message_tokens(messages: list[BaseMessage]) -> int:
    """Count tokens with the framework counter (no custom bookkeeping)."""
    return int(count_tokens_approximately(messages))


def ensure_message_ids(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Ensure every message carries an id (required by LangMem)."""
    import uuid

    for message in messages:
        if not getattr(message, "id", None):
            try:
                message.id = f"msg-{uuid.uuid4().hex[:12]}"
            except Exception:  # pragma: no cover - defensive
                pass
    return messages


def needs_compaction(messages: list[BaseMessage], *, trigger_tokens: int) -> bool:
    """Token-driven compaction decision; never fixed-turn-count driven."""
    return count_message_tokens(messages) >= trigger_tokens


def running_summary_from_state(
    *, summary_text: str, context: dict[str, Any] | None
) -> RunningSummary | None:
    """Recover the persisted LangMem running summary without overwriting it.

    A caller default of ``""`` never clears persisted state: when the stored
    context already holds a running summary it is preserved verbatim.
    """
    context = context or {}
    stored = context.get("running_summary")
    if isinstance(stored, RunningSummary):
        return stored
    text = summary_text.strip()
    if not text:
        return None
    return RunningSummary(
        summary=text, summarized_message_ids=set(), last_summarized_message_id=None
    )


def summary_text_from_running(running: RunningSummary | None) -> str:
    """Extract the continuity text from a LangMem running summary."""
    if running is None:
        return ""
    return str(running.summary)


__all__ = [
    "CHECKPOINT_ENV_VAR",
    "CHECKPOINT_FILENAME",
    "COMPACTION_TRIGGER_TOKENS",
    "CONTEXT_REFERENCE_TOKENS",
    "MAX_SUMMARY_TOKENS",
    "RECENT_KEEP_TOKENS",
    "CheckpointerFactory",
    "MemoryConfig",
    "SqliteCheckpointerFactory",
    "build_summarization_node",
    "count_message_tokens",
    "default_checkpoint_dir",
    "default_memory_config",
    "ensure_message_ids",
    "needs_compaction",
    "running_summary_from_state",
    "summary_text_from_running",
    "thread_id_for_chat",
]
