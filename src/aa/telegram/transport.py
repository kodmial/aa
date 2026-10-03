"""Telegram transport boundary.

Implements the Telegram Bot API long-polling transport with the standard
library only (no third-party Telegram/HTTP clients):

- startup bootstrap: ``getMe`` (fail closed) -> ``deleteWebhook`` ->
  ``setMyCommands`` -> description/short-description (best effort) ->
  ``getUpdates`` long polling;
- private text messages only;
- ``/start`` and ``/new`` command recognition with a routing hook;
- ``getUpdates`` offset handling and duplicate/idempotency handling;
- ``sendMessage`` delivery with bounded retry/backoff;
- graceful cancellation/shutdown;
- privacy-safe operational logs (no bot token, no raw message bodies).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import random
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("aa.telegram.transport")

DEFAULT_BASE_URL = "https://api.telegram.org"
DEFAULT_POLL_TIMEOUT_SECONDS = 25
DEFAULT_POLL_LIMIT = 100

SUPPORTED_COMMANDS: tuple[tuple[str, str], ...] = (
    ("start", "Start the bot"),
    ("new", "Start a new session"),
)

BOT_DESCRIPTION = "AA support bot over Telegram long polling."
BOT_SHORT_DESCRIPTION = "AA support bot"

RECOGNIZED_COMMANDS = frozenset({"start", "new"})


class TelegramApiError(Exception):
    """Transient or permanent Telegram Bot API failure."""


class TelegramAuthError(TelegramApiError):
    """Invalid/revoked bot token. Startup must fail closed."""


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


@dataclass(frozen=True)
class TelegramIncoming:
    """Parsed private text message with command routing info."""

    update_id: int
    chat_id: int
    message_id: int
    text: str
    command: str | None = None


UpdateHandler = Callable[[TelegramIncoming], Awaitable[None]]


def parse_command(text: str) -> str | None:
    """Extract a bot command name from message text, if present."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    token = stripped[1:].split(None, 1)[0] if len(stripped) > 1 else ""
    if not token:
        return None
    # Strip ``@BotName`` suffix form (``/start@MyBot``).
    command = token.split("@", 1)[0].strip().lower()
    if not command:
        return None
    return command


def parse_update(raw: Any) -> TelegramIncoming | None:
    """Parse one raw ``getUpdates`` entry into private text, else ``None``."""
    if not isinstance(raw, dict):
        return None
    update_id = raw.get("update_id")
    message = raw.get("message")
    if not isinstance(update_id, int) or not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    if chat.get("type") != "private":
        return None
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    text = message.get("text")
    if not isinstance(chat_id, int) or not isinstance(message_id, int):
        return None
    if not isinstance(text, str) or not text:
        return None
    command = parse_command(text)
    return TelegramIncoming(
        update_id=update_id,
        chat_id=chat_id,
        message_id=message_id,
        text=text,
        command=command,
    )


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


class TelegramApi(ABC):
    """Minimal async Telegram Bot API surface used by the transport."""

    @abstractmethod
    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        """Call a Bot API method and return the ``result`` payload."""
        raise NotImplementedError


class UrllibTelegramApi(TelegramApi):
    """Stdlib ``urllib``-backed Bot API client (outbound HTTPS only)."""

    def __init__(
        self,
        token: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 40.0,
    ) -> None:
        if not token:
            raise ValueError("Telegram bot token is required")
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        return await asyncio.to_thread(self._call_sync, method, payload)

    def _call_sync(self, method: str, payload: dict[str, Any]) -> Any:
        url = f"{self._base_url}/bot{self._token}/{method}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise _translate_http_error(exc) from exc
        except OSError as exc:
            raise TelegramApiError(f"telegram {method} network error") from exc
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TelegramApiError(f"telegram {method} bad response") from exc
        return _translate_envelope(method, envelope)


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _translate_http_error(exc: urllib.error.HTTPError) -> TelegramApiError:
    body = _read_http_error_body(exc)
    description = ""
    error_code: int | None = None
    if body:
        try:
            envelope = json.loads(body)
            if isinstance(envelope, dict):
                desc = envelope.get("description")
                code = envelope.get("error_code")
                if isinstance(desc, str):
                    description = desc
                if isinstance(code, int):
                    error_code = code
        except json.JSONDecodeError:
            description = ""
    code = error_code if error_code is not None else exc.code
    if code == 401 or "unauthorized" in description.lower():
        return TelegramAuthError("telegram unauthorized: invalid bot token")
    suffix = f": {description}" if description else ""
    return TelegramApiError(f"telegram http error {code}{suffix}")


def _translate_envelope(method: str, envelope: Any) -> Any:
    if not isinstance(envelope, dict):
        raise TelegramApiError(f"telegram {method} bad response")
    if not envelope.get("ok"):
        error_code = envelope.get("error_code")
        description = envelope.get("description", "")
        if error_code == 401 or (
            isinstance(description, str) and "unauthorized" in description.lower()
        ):
            raise TelegramAuthError("telegram unauthorized: invalid bot token")
        raise TelegramApiError(f"telegram {method} failed: {description}")
    return envelope.get("result")


class PollingTelegramTransport(TelegramTransport):
    """Outbound long-polling transport (no inbound listener)."""

    def __init__(
        self,
        token: str,
        *,
        api: TelegramApi | None = None,
        base_url: str = DEFAULT_BASE_URL,
        update_handler: UpdateHandler | None = None,
        poll_timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
        poll_limit: int = DEFAULT_POLL_LIMIT,
        max_bootstrap_retries: int = 3,
        max_send_retries: int = 4,
        retry_base_delay_seconds: float = 0.2,
        retry_max_delay_seconds: float = 5.0,
        seen_capacity: int = 5000,
        drop_pending_updates: bool = False,
    ) -> None:
        if not token:
            raise ValueError("Telegram bot token is required")
        if poll_limit <= 0:
            raise ValueError("poll_limit must be > 0")
        if seen_capacity <= 0:
            raise ValueError("seen_capacity must be > 0")
        self._token_present = True
        self._api: TelegramApi = (
            api if api is not None else UrllibTelegramApi(token, base_url=base_url)
        )
        self._update_handler = update_handler
        self._command_handlers: dict[str, UpdateHandler] = {}
        self._poll_timeout_seconds = poll_timeout_seconds
        self._poll_limit = poll_limit
        self._max_bootstrap_retries = max_bootstrap_retries
        self._max_send_retries = max_send_retries
        self._retry_base_delay = retry_base_delay_seconds
        self._retry_max_delay = retry_max_delay_seconds
        self._seen_capacity = seen_capacity
        self._drop_pending_updates = drop_pending_updates
        self._seen_ids: set[int] = set()
        self._seen_order: collections.deque[int] = collections.deque()
        self._offset: int | None = None
        self._received: list[TelegramIncoming] = []
        self._sent: list[TelegramReply] = []
        self._bot_info: dict[str, Any] | None = None
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._consecutive_poll_failures = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def offset(self) -> int | None:
        """Next ``getUpdates`` offset (``None`` before the first batch)."""
        return self._offset

    @property
    def received(self) -> list[TelegramIncoming]:
        """Parsed inbound messages in arrival order (for tests/routing)."""
        return list(self._received)

    @property
    def sent_messages(self) -> list[TelegramReply]:
        """Replies delivered via ``send`` (for tests)."""
        return list(self._sent)

    @property
    def bot_info(self) -> dict[str, Any] | None:
        """Verified ``getMe`` result after a successful bootstrap."""
        return dict(self._bot_info) if self._bot_info is not None else None

    def on_update(self, handler: UpdateHandler) -> None:
        """Set the generic inbound-message handler."""
        self._update_handler = handler

    def on_command(self, command: str, handler: UpdateHandler) -> None:
        """Register a routing hook for one command (e.g. ``new``)."""
        name = command.strip().lstrip("/").split("@", 1)[0].strip().lower()
        if not name:
            raise ValueError("command name must not be empty")
        self._command_handlers[name] = handler

    async def start(self) -> None:
        """Perform the startup contract then start long polling."""
        if self._running:
            return
        if not self._token_present:
            raise TelegramAuthError("telegram unauthorized: invalid bot token")
        self._stop_event.clear()
        await self._bootstrap()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop(), name="telegram-polling")
        logger.info("telegram polling started")

    async def stop(self) -> None:
        """Stop polling without losing an update that was already handled.

        Prefer allowing the in-flight long poll to return naturally. This
        avoids leaving a cancelled urllib worker thread issuing getUpdates in
        parallel with shutdown acknowledgement.
        """
        self._stop_event.set()
        task = self._poll_task
        self._poll_task = None
        clean_shutdown = True
        if task is not None:
            try:
                # Give a handler / short poll a chance to finish, but keep
                # operator-triggered stop responsive.
                await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
            except TimeoutError:
                clean_shutdown = False
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                clean_shutdown = False
            except Exception:
                logger.warning("telegram polling task ended with error")
        if clean_shutdown:
            await self._acknowledge_offset()
        self._running = False
        logger.info("telegram polling stopped")

    async def send(self, reply: TelegramReply) -> None:
        """Deliver one reply via ``sendMessage`` with bounded retry."""
        payload: dict[str, Any] = {"chat_id": reply.chat_id, "text": reply.text}
        await self._call_with_retry("sendMessage", payload, max_retries=self._max_send_retries)
        self._sent.append(reply)
        logger.info(
            "telegram message sent",
            extra={"chat_id": reply.chat_id, "text_len": len(reply.text)},
        )

    async def _bootstrap(self) -> None:
        me = await self._call_with_retry("getMe", {}, max_retries=self._max_bootstrap_retries)
        if not isinstance(me, dict) or not me.get("id"):
            raise TelegramApiError("telegram getMe returned an unexpected result")
        raw_username = me.get("username")
        username: str = raw_username if isinstance(raw_username, str) else ""
        self._bot_info = dict(me)
        logger.info(
            "telegram bootstrap verified bot",
            extra={"bot_id": me.get("id"), "username_len": len(username)},
        )
        await self._call_with_retry(
            "deleteWebhook",
            {"drop_pending_updates": self._drop_pending_updates},
            max_retries=self._max_bootstrap_retries,
        )
        await self._call_with_retry(
            "setMyCommands",
            {
                "commands": [
                    {"command": name, "description": desc} for name, desc in SUPPORTED_COMMANDS
                ]
            },
            max_retries=self._max_bootstrap_retries,
        )
        # Description fields are best effort: older/server variants may not
        # support them, so failures here must not fail startup. A single
        # attempt each avoids amplifying "unsupported method" into retries.
        for method, payload in (
            ("setMyDescription", {"description": BOT_DESCRIPTION}),
            ("setMyShortDescription", {"short_description": BOT_SHORT_DESCRIPTION}),
        ):
            try:
                await self._api.call(method, dict(payload))
            except TelegramAuthError:
                raise
            except (TelegramApiError, TimeoutError, OSError):
                logger.warning("telegram bootstrap optional step skipped", extra={"method": method})

    async def _call_with_retry(
        self, method: str, payload: dict[str, Any], *, max_retries: int
    ) -> Any:
        attempt = 0
        while True:
            try:
                return await self._api.call(method, dict(payload))
            except TelegramAuthError:
                raise
            except (TelegramApiError, TimeoutError, OSError) as exc:
                attempt += 1
                if attempt > max_retries:
                    logger.warning(
                        "telegram call failed",
                        extra={"method": method, "attempts": attempt},
                    )
                    raise
                delay = self._backoff_delay(attempt)
                logger.info(
                    "telegram call retrying",
                    extra={"method": method, "attempt": attempt, "delay": delay},
                )
                await asyncio.sleep(delay)
                _ = exc

    def _backoff_delay(self, attempt: int) -> float:
        capped: float = min(self._retry_max_delay, self._retry_base_delay * (2 ** (attempt - 1)))
        jitter: float = random.uniform(0, capped * 0.2)
        return min(self._retry_max_delay, capped + jitter)

    async def _poll_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    batch = await self._fetch_updates()
                except asyncio.CancelledError:
                    break
                except TelegramAuthError:
                    logger.warning("telegram polling unauthorized; stopping")
                    break
                except Exception:
                    self._consecutive_poll_failures += 1
                    delay = self._backoff_delay(min(self._consecutive_poll_failures, 6))
                    logger.info(
                        "telegram poll retrying",
                        extra={"failures": self._consecutive_poll_failures},
                    )
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                    except TimeoutError:
                        pass
                    except asyncio.CancelledError:
                        break
                    continue
                self._consecutive_poll_failures = 0
                for raw in batch:
                    if self._stop_event.is_set():
                        break
                    handled = await self._process_raw_update(raw)
                    if not handled:
                        # Do not advance past a failed update. Telegram will
                        # redeliver it because the offset is committed only
                        # after successful handling.
                        try:
                            await asyncio.wait_for(
                                self._stop_event.wait(),
                                timeout=self._backoff_delay(1),
                            )
                        except TimeoutError:
                            pass
                        break
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    async def _fetch_updates(self) -> list[Any]:
        payload: dict[str, Any] = {
            "limit": self._poll_limit,
            "timeout": self._poll_timeout_seconds,
            "allowed_updates": ["message"],
        }
        if self._offset is not None:
            payload["offset"] = self._offset
        result = await self._api.call("getUpdates", payload)
        if result is None:
            return []
        if not isinstance(result, list):
            raise TelegramApiError("telegram getUpdates returned an unexpected result")
        return result

    def _is_duplicate(self, update_id: int) -> bool:
        return update_id in self._seen_ids

    def _commit_update_id(self, update_id: int) -> None:
        """Mark one update handled and advance the next polling offset."""
        if update_id not in self._seen_ids:
            self._seen_ids.add(update_id)
            self._seen_order.append(update_id)
            while len(self._seen_order) > self._seen_capacity:
                oldest = self._seen_order.popleft()
                self._seen_ids.discard(oldest)
        next_offset = update_id + 1
        self._offset = next_offset if self._offset is None else max(self._offset, next_offset)

    async def _acknowledge_offset(self) -> None:
        """Best-effort Telegram acknowledgement for already handled updates."""
        if self._offset is None:
            return
        payload: dict[str, Any] = {
            "offset": self._offset,
            "limit": 1,
            "timeout": 0,
            "allowed_updates": ["message"],
        }
        try:
            await self._api.call("getUpdates", payload)
        except TelegramAuthError:
            logger.warning("telegram final acknowledgement unauthorized")
        except (TelegramApiError, TimeoutError, OSError):
            logger.warning("telegram final acknowledgement failed")

    async def _process_raw_update(self, raw: Any) -> bool:
        update_id = raw.get("update_id") if isinstance(raw, dict) else None
        if not isinstance(update_id, int):
            logger.info("telegram update without id skipped")
            return True
        if self._is_duplicate(update_id):
            self._commit_update_id(update_id)
            logger.info("telegram duplicate update skipped", extra={"update_id": update_id})
            return True

        parsed = parse_update(raw)
        if parsed is None:
            # Unsupported update types are intentionally consumed so they do
            # not poison the polling queue forever.
            self._commit_update_id(update_id)
            return True

        logger.info(
            "telegram update received",
            extra={
                "update_id": parsed.update_id,
                "chat_id": parsed.chat_id,
                "message_id": parsed.message_id,
                "text_len": len(parsed.text),
                "command": parsed.command or "",
            },
        )
        if not await self._dispatch(parsed):
            return False
        self._received.append(parsed)
        self._commit_update_id(update_id)
        return True

    async def _dispatch(self, incoming: TelegramIncoming) -> bool:
        handler: UpdateHandler | None = None
        if incoming.command is not None:
            handler = self._command_handlers.get(incoming.command)
        if handler is None:
            handler = self._update_handler
        if handler is None:
            return True
        try:
            await handler(incoming)
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "telegram update handler failed",
                extra={
                    "update_id": incoming.update_id,
                    "chat_id": incoming.chat_id,
                    "command": incoming.command or "",
                },
            )
            return False
