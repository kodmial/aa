"""Per-chat session coordination.

Tracks one logical session per Telegram chat so later issues can route
updates to the correct OpenCode session. Pure in-memory bookkeeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ChatSession:
    """State for a single chat."""

    chat_id: int
    created_at: float = field(default_factory=time.monotonic)
    message_count: int = 0


class SessionCoordinator:
    """Owns the mapping from chat id to :class:`ChatSession`."""

    def __init__(self) -> None:
        self._sessions: dict[int, ChatSession] = {}
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

    def record_message(self, chat_id: int) -> ChatSession:
        """Record one inbound message for ``chat_id``."""
        session = self.get_or_create(chat_id)
        session.message_count += 1
        return session

    def session_count(self) -> int:
        """Return the number of tracked chats."""
        return len(self._sessions)
