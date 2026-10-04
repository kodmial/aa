"""Per-chat session coordination.

Tracks one logical session per Telegram chat and binds each chat to a
distinct OpenCode session identity. The OpenCode server owns conversation
durability (see ``docs/opencode-runtime.md``); the worker only keeps the
``chat_id -> opencode session id`` mapping in memory plus a reset
generation counter, so no external state service is needed.

Reset semantics: forgetting the old OpenCode session id and creating a new
session guarantees the next turn starts a fresh conversation. The old
remote session is deleted best-effort; if it is already gone the reset
still succeeds.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from aa.opencode.client import OpenCodeClient
from aa.opencode.errors import OpenCodeSessionNotFoundError

logger = logging.getLogger("aa.sessions.coordinator")


@dataclass
class ChatSession:
    """State for a single chat."""

    chat_id: int
    created_at: float = field(default_factory=time.monotonic)
    message_count: int = 0
    opencode_session_id: str | None = None
    generation: int = 0


class SessionCoordinator:
    """Owns the mapping from chat id to :class:`ChatSession`.

    Concurrency (issue #5): session create/rebind/reset for the same chat
    are serialized through a per-chat asyncio lock, so two simultaneous
    first messages for one chat cannot create competing OpenCode sessions.
    Different chats use different locks and never block each other here.
    The lock body contains no ``await`` between the map lookup and the lock
    creation, so lock creation itself is atomic on the single event loop.
    """

    def __init__(self) -> None:
        self._sessions: dict[int, ChatSession] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._running = False

    async def start(self) -> None:
        """Enable coordination."""
        self._running = True

    async def stop(self) -> None:
        """Disable coordination (sessions are kept for inspection)."""
        self._running = False

    @property
    def running(self) -> bool:
        """Whether the coordinator is running."""
        return self._running

    def get_or_create(self, chat_id: int) -> ChatSession:
        """Return the existing session for ``chat_id`` or create one."""
        session = self._sessions.get(chat_id)
        if session is None:
            session = ChatSession(chat_id=chat_id)
            self._sessions[chat_id] = session
        return session

    def _lock_for(self, chat_id: int) -> asyncio.Lock:
        """Return the serialization lock for ``chat_id`` (creating it)."""
        existing = self._locks.get(chat_id)
        if existing is not None:
            return existing
        created = asyncio.Lock()
        self._locks[chat_id] = created
        return created

    def record_message(self, chat_id: int) -> ChatSession:
        """Record one inbound message for ``chat_id``."""
        session = self.get_or_create(chat_id)
        session.message_count += 1
        return session

    def session_count(self) -> int:
        """Return the number of tracked chats."""
        return len(self._sessions)

    def get_opencode_session_id(self, chat_id: int) -> str | None:
        """Return the bound OpenCode session id for ``chat_id``, if any."""
        session = self._sessions.get(chat_id)
        return session.opencode_session_id if session is not None else None

    def set_opencode_session_id(self, chat_id: int, opencode_session_id: str) -> ChatSession:
        """Bind ``chat_id`` to an existing OpenCode session id."""
        if not opencode_session_id:
            raise ValueError("opencode_session_id must not be empty")
        session = self.get_or_create(chat_id)
        session.opencode_session_id = opencode_session_id
        return session

    def reset(self, chat_id: int) -> ChatSession:
        """Forget the OpenCode binding for ``chat_id`` and start over.

        The message counter restarts and the generation counter advances so
        operators can tell a reset session apart from a fresh chat. Only
        structured ids are logged, never message content.
        """
        session = self.get_or_create(chat_id)
        session.opencode_session_id = None
        session.message_count = 0
        session.generation += 1
        logger.info("session reset")
        return session

    async def ensure_opencode_session(
        self, chat_id: int, client: OpenCodeClient, title: str = ""
    ) -> str:
        """Return the OpenCode session id for ``chat_id``, creating it.

        Reuses the bound session when present (continuation); otherwise
        creates a new remote session and binds it. Distinct chats always
        receive distinct OpenCode sessions because each binding is stored
        under its own chat id. The whole check-and-create is serialized per
        chat so concurrent first messages cannot create competing sessions.
        """
        async with self._lock_for(chat_id):
            return await self._ensure_unlocked(chat_id, client, title)

    def _ensure_unlocked_sync(self, chat_id: int) -> str | None:
        session = self._sessions.get(chat_id)
        return session.opencode_session_id if session is not None else None

    async def _ensure_unlocked(self, chat_id: int, client: OpenCodeClient, title: str = "") -> str:
        existing = self._ensure_unlocked_sync(chat_id)
        if existing is not None:
            return existing
        info = await client.create_session(title or f"chat-{chat_id}")
        self.set_opencode_session_id(chat_id, info.id)
        return info.id

    async def reset_opencode_session(
        self, chat_id: int, client: OpenCodeClient, *, delete_remote: bool = True
    ) -> str:
        """Reset the conversation for ``chat_id`` and return the new id.

        The previous remote session is deleted best-effort (a missing
        session is not an error); the mapping is then cleared and a fresh
        session is created and bound. Serialized per chat so a reset can
        never interleave with a concurrent create/rebind for the same chat;
        callers must additionally order resets against in-flight turns via
        the per-chat dispatcher (issue #5) so a reset never retires a
        session while an older turn is still mutating it.
        """
        async with self._lock_for(chat_id):
            previous = self._ensure_unlocked_sync(chat_id)
            if previous is not None and delete_remote:
                try:
                    await client.delete_session(previous)
                except OpenCodeSessionNotFoundError:
                    pass
            self.reset(chat_id)
            return await self._ensure_unlocked(chat_id, client)
