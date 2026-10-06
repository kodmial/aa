"""Continuous Telegram typing heartbeat for ordinary turns (issue #118).

Contract (Product Implementation #112 UI):

- start ``sendChatAction(chat_id, "typing")`` immediately after an accepted
  normal user turn begins processing;
- heartbeat independently of planner/retrieval/generation;
- refresh at approximately 4-second cadence (configurable);
- keep typing active across model retries, retrieval, grounding repair and
  ``sendMessage``/``sendVoice`` preparation;
- do not stop when generation finishes;
- stop only after Telegram confirms final outbound delivery succeeded, or
  the turn is definitively aborted with no outbound message;
- retrying delivery keeps the heartbeat alive;
- cancellation/new-session/shutdown cleanly cancels the heartbeat task.

No avoidable ``typing disappeared -> dead gap -> message arrives``
transition: the heartbeat keeps firing until delivery confirmation.

Privacy: only chat-action counts and failure categories are logged, never
message bodies or raw identifiers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

logger = logging.getLogger("aa.telegram.typing")

DEFAULT_TYPING_INTERVAL_SECONDS = 4.0


class ChatActionSender(Protocol):
    """Minimal transport surface needed by the heartbeat."""

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Send one ``sendChatAction`` event (must not raise on success)."""
        raise NotImplementedError


class TypingHeartbeat:
    """Independent asyncio heartbeat keeping Telegram typing visible.

    One instance owns exactly one turn's heartbeat task. :meth:`start`
    fires the first ``typing`` action immediately, then refreshes on the
    configured cadence until :meth:`stop` is called after confirmed
    delivery (or definitive abort). Retried delivery keeps the same
    instance alive; callers must not stop it between generation and
    delivery.
    """

    def __init__(
        self,
        sender: ChatActionSender,
        chat_id: int,
        *,
        interval_seconds: float = DEFAULT_TYPING_INTERVAL_SECONDS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self._sender = sender
        self._chat_id = chat_id
        self._interval = float(interval_seconds)
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._stopped = False
        self.sends = 0

    @property
    def running(self) -> bool:
        """Whether the heartbeat task is currently active."""
        return self._task is not None and not self._task.done()

    @property
    def interval_seconds(self) -> float:
        """Configured refresh cadence."""
        return self._interval

    async def start(self) -> None:
        """Fire the first typing action immediately and arm the loop."""
        if self._started:
            return
        self._started = True
        self._stopped = False
        await self._fire_once()
        self._task = asyncio.create_task(self._loop(), name="aa-typing-heartbeat")

    async def stop(self) -> None:
        """Cancel the heartbeat loop (idempotent, no delivery I/O)."""
        self._stopped = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _fire_once(self) -> None:
        try:
            await self._sender.send_chat_action(self._chat_id, "typing")
            self.sends += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.info("typing heartbeat send failed")

    async def _loop(self) -> None:
        try:
            while not self._stopped:
                try:
                    await asyncio.sleep(self._interval)
                except asyncio.CancelledError:
                    break
                if self._stopped:
                    break
                await self._fire_once()
        except asyncio.CancelledError:
            pass

    async def __aenter__(self) -> TypingHeartbeat:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()


__all__ = [
    "DEFAULT_TYPING_INTERVAL_SECONDS",
    "ChatActionSender",
    "TypingHeartbeat",
]
