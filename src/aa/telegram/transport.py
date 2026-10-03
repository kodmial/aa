"""Telegram transport boundary.

This module defines transport-neutral interfaces for receiving Telegram
updates and sending replies. No live Telegram network calls are made here;
concrete polling/webhook transports will be added by later issues.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TelegramUpdate:
    """A minimal inbound chat update (transport-neutral)."""

    chat_id: int
    message_id: int
    text: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TelegramReply:
    """A minimal outbound reply (transport-neutral)."""

    chat_id: int
    text: str


class TelegramTransport(ABC):
    """Interface every Telegram transport must implement."""

    @abstractmethod
    async def start(self) -> None:
        """Start receiving updates."""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Stop receiving updates and release resources."""
        raise NotImplementedError

    @abstractmethod
    async def send(self, reply: TelegramReply) -> None:
        """Queue or deliver an outbound reply."""
        raise NotImplementedError

    @property
    @abstractmethod
    def running(self) -> bool:
        """Whether the transport is currently running."""
        raise NotImplementedError


class StubTelegramTransport(TelegramTransport):
    """In-memory transport used for lifecycle wiring without network I/O."""

    def __init__(self) -> None:
        self._running = False
        self.sent: list[TelegramReply] = []

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send(self, reply: TelegramReply) -> None:
        self.sent.append(reply)

    @property
    def running(self) -> bool:
        return self._running
