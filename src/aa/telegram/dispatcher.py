"""Keyed per-chat turn dispatcher (issue #5).

One authoritative Telegram poller and one local ``opencode serve`` process
exist per AA worker; isolation is by opaque OpenCode session identity, not
by one OpenCode process per Telegram user.

Concurrency contract implemented here:

- turns for the same chat are strict FIFO with at most one active turn;
- turns for different chats may execute concurrently;
- a configurable global ``MAX_CONCURRENT_TURNS`` semaphore bounds total
  concurrent turns so the worker cannot overload OpenCode/provider/runtime;
- each per-chat pending queue is bounded (``per_chat_queue_size``); overflow
  backpressures safely instead of growing memory without bound;
- session create/rebind/reset for a chat runs inside that chat's serialized
  worker, so two simultaneous first messages cannot create competing
  sessions and ``/new`` is ordered relative to ordinary turns for that
  chat and never resets a session while an older turn is still mutating it;
- one slow chat never head-of-line blocks unrelated chats: the poller
  callback only enqueues (fast) and returns, while per-chat workers run as
  independent asyncio tasks guarded only by the global semaphore.

Restart/crash policy: conversation continuity is guaranteed only for the
lifetime of the authoritative AA runtime. The dispatcher keeps no durable
state: Telegram offset/idempotency is owned by the transport and the
chat->session map by :class:`SessionCoordinator`. An offset is committed by
the transport once the update is accepted into the dispatcher queue. The
explicit crash window is therefore: an update accepted but not yet processed
when the worker crashes is lost (at-most-once for in-flight turns) and will
not be redelivered, because no durable transactional queue exists. Clean
shutdown drains: :meth:`ChatTurnDispatcher.stop` waits for in-flight turns
and (bounded) for queued turns before releasing resources. If strict
cross-crash exactly-once reply semantics become a product requirement, a
small durable idempotency store must be added as a separate measured
capability rather than coupling it to OpenCode session history.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from aa.telegram.transport import TelegramIncoming

logger = logging.getLogger("aa.telegram.dispatcher")

TurnProcessor = Callable[[TelegramIncoming], Awaitable[None]]


class ChatQueueFullError(Exception):
    """Raised when a per-chat pending queue is at its configured bound."""


@dataclass
class _ChatWorker:
    """Serial worker state for one Telegram chat."""

    chat_id: int
    queue: asyncio.Queue[TelegramIncoming] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None


class ChatTurnDispatcher:
    """Per-chat FIFO dispatcher with a global concurrency bound.

    The poller thread calls :meth:`submit` (fast, non-blocking); per-chat
    worker tasks call ``process`` sequentially; a shared semaphore bounds
    how many workers may be inside ``process`` at once.
    """

    def __init__(
        self,
        process: TurnProcessor,
        *,
        max_concurrent_turns: int,
        per_chat_queue_size: int,
    ) -> None:
        if max_concurrent_turns <= 0:
            raise ValueError("max_concurrent_turns must be > 0")
        if per_chat_queue_size <= 0:
            raise ValueError("per_chat_queue_size must be > 0")
        self._process = process
        self._max_concurrent_turns = max_concurrent_turns
        self._per_chat_queue_size = per_chat_queue_size
        self._semaphore = asyncio.Semaphore(max_concurrent_turns)
        self._workers: dict[int, _ChatWorker] = {}
        self._running = False

    @property
    def running(self) -> bool:
        """Whether the dispatcher accepts and processes turns."""
        return self._running

    @property
    def max_concurrent_turns(self) -> int:
        """Configured global concurrency bound."""
        return self._max_concurrent_turns

    @property
    def per_chat_queue_size(self) -> int:
        """Configured per-chat pending queue bound."""
        return self._per_chat_queue_size

    def pending_count(self, chat_id: int) -> int:
        """Return the queued (not yet active) turn count for ``chat_id``."""
        worker = self._workers.get(chat_id)
        return worker.queue.qsize() if worker is not None else 0

    def chat_count(self) -> int:
        """Return the number of chats with worker state."""
        return len(self._workers)

    async def start(self) -> None:
        """Enable dispatching."""
        self._running = True

    async def stop(self, *, drain_timeout: float = 5.0) -> None:
        """Release worker tasks after a bounded drain on clean shutdown.

        New submits are rejected first; queued and in-flight turns are given
        ``drain_timeout`` seconds to finish so clean shutdown/handoff stays
        safe. Anything still pending after the deadline is dropped: that is
        the documented crash window, not exactly-once delivery.
        """
        self._running = False
        workers = list(self._workers.values())
        if workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(worker.queue.join() for worker in workers)),
                    timeout=drain_timeout,
                )
            except TimeoutError:
                logger.warning("dispatcher drain timed out; dropping queued turns")
        self._workers.clear()
        for worker in workers:
            if worker.task is not None:
                worker.task.cancel()
        for worker in workers:
            if worker.task is not None:
                try:
                    await worker.task
                except asyncio.CancelledError:
                    pass

    async def submit(self, incoming: TelegramIncoming) -> None:
        """Enqueue one accepted update for its chat (fast, non-blocking).

        Raises :class:`ChatQueueFullError` when that chat's pending queue is
        at its configured bound; the caller must backpressure safely (single
        bounded reply, update still acknowledged) rather than growing memory.
        Raises ``RuntimeError`` when the dispatcher is not running.
        """
        if not self._running:
            raise RuntimeError("dispatcher is not running")
        worker = self._workers.get(incoming.chat_id)
        if worker is None:
            worker = _ChatWorker(chat_id=incoming.chat_id)
            worker.task = asyncio.create_task(
                self._run_chat(worker), name=f"aa-chat-{incoming.chat_id}"
            )
            self._workers[incoming.chat_id] = worker
        if worker.queue.qsize() >= self._per_chat_queue_size:
            raise ChatQueueFullError(f"chat queue is full for chat {incoming.chat_id}")
        worker.queue.put_nowait(incoming)

    async def _run_chat(self, worker: _ChatWorker) -> None:
        """Serialize one chat's turns FIFO, bounded globally by semaphore."""
        try:
            while True:
                try:
                    incoming = await worker.queue.get()
                except asyncio.CancelledError:
                    break
                try:
                    async with self._semaphore:
                        try:
                            await self._process(incoming)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.warning(
                                "chat turn failed",
                                extra={"chat_id": worker.chat_id},
                            )
                except asyncio.CancelledError:
                    worker.queue.task_done()
                    break
                except Exception:
                    logger.warning(
                        "chat turn failed",
                        extra={"chat_id": worker.chat_id},
                    )
                finally:
                    try:
                        worker.queue.task_done()
                    except ValueError:
                        pass
                if not self._running and worker.queue.empty():
                    break
        except asyncio.CancelledError:
            pass
        finally:
            current = self._workers.get(worker.chat_id)
            if current is worker and (not self._running or worker.queue.empty()):
                self._workers.pop(worker.chat_id, None)


__all__ = ["ChatQueueFullError", "ChatTurnDispatcher", "TurnProcessor"]
