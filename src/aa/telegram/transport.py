"""Telegram transport boundary.

Implements the Telegram Bot API long-polling transport with the standard
library only (no third-party Telegram/HTTP clients):

- startup bootstrap: ``getMe`` (fail closed) -> ``deleteWebhook`` ->
  ``setMyCommands`` -> description/short-description (best effort) ->
  ``getUpdates`` long polling;
- private text and voice messages only;
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

# Authoritative user-facing output envelope (issue #83). The transport is
# the final defense-in-depth guard: it never splits overflow into multiple
# messages and never logs response text (lengths/categories only).
TELEGRAM_HARD_CHARS = 900
TELEGRAM_HARD_WORDS = 130


def _check_outbound_envelope(text: str) -> None:
    """Fail closed when ``text`` exceeds the hard Telegram envelope.

    Raises :class:`TelegramApiError` before any network I/O, without
    including or logging the response text. Overflow is never split.
    """
    from aa.conversation.output_limits import check_envelope

    result = check_envelope(text)
    if result["passed"]:
        return
    logger.warning(
        "telegram outbound envelope violation blocked",
        extra={
            "category": result["category"],
            "graphemes": result["graphemes"],
            "words": result["words"],
            "quoted": result["quoted"],
        },
    )
    raise TelegramEnvelopeError(f"telegram reply exceeds output envelope [{result['category']}]")


SUPPORTED_COMMANDS: tuple[tuple[str, str], ...] = (
    ("start", "Start the bot"),
    ("new", "Start a new session"),
)

BOT_DESCRIPTION = "AA support bot over Telegram long polling."
BOT_SHORT_DESCRIPTION = "AA support bot"

RECOGNIZED_COMMANDS = frozenset({"start", "new"})


class TelegramApiError(Exception):
    """Transient or permanent Telegram Bot API failure."""


class TelegramEnvelopeError(TelegramApiError):
    """Outbound reply blocked by the hard envelope guard (no network I/O)."""


class TelegramAuthError(TelegramApiError):
    """Invalid/revoked bot token. Startup must fail closed."""


class TelegramConflictError(TelegramApiError):
    """Another poller/webhook owns getUpdates (HTTP 409). Startup blocked."""


class TelegramRateLimitedError(TelegramApiError):
    """Telegram reports too many requests (HTTP 429). Back off and retry."""


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
    reply_markup: dict[str, Any] | None = None
    message_id: int | None = None


@dataclass(frozen=True)
class TelegramVoiceReply:
    """An outbound voice reply carrying OGG/Opus bytes for ``sendVoice``."""

    chat_id: int
    voice_bytes: bytes
    reply_markup: dict[str, Any] | None = None
    caption: str | None = None


@dataclass(frozen=True)
class VoiceAttachment:
    """A Telegram voice note reference (audio fetched separately)."""

    file_id: str
    duration_seconds: int
    file_size_bytes: int | None = None


@dataclass(frozen=True)
class TelegramIncoming:
    """Parsed private message with command routing info.

    Text messages carry ``text``; voice notes carry ``voice`` with an
    empty ``text``. Callback button presses carry ``is_callback`` with
    ``callback_data`` and an empty ``text``. Text parsing is unchanged
    by voice/callback support.
    """

    update_id: int
    chat_id: int
    message_id: int
    text: str
    command: str | None = None
    voice: VoiceAttachment | None = None
    is_callback: bool = False
    callback_id: str | None = None
    callback_data: str | None = None
    sender_id: int | None = None
    callback_message_id: int | None = None
    callback_inaccessible: bool = False


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


def parse_callback_query(raw: Any) -> TelegramIncoming | None:
    """Parse one raw ``callback_query`` entry into a typed callback, else ``None``."""
    if not isinstance(raw, dict):
        return None
    update_id = raw.get("update_id")
    query = raw.get("callback_query")
    if not isinstance(update_id, int) or not isinstance(query, dict):
        return None
    callback_id = query.get("id")
    sender = query.get("from")
    data = query.get("data")
    if not isinstance(callback_id, str) or not callback_id:
        return None
    if not isinstance(sender, dict) or not isinstance(sender.get("id"), int):
        return None
    if not isinstance(data, str) or not data:
        return None
    sender_id = int(sender["id"])
    message = query.get("message")
    if isinstance(message, dict):
        chat = message.get("chat")
        chat_id = chat.get("id") if isinstance(chat, dict) else None
        bound_message_id = message.get("message_id")
        if not isinstance(chat_id, int) or not isinstance(bound_message_id, int):
            return None
        if isinstance(chat, dict) and chat.get("type") != "private":
            return None
        return TelegramIncoming(
            update_id=update_id,
            chat_id=chat_id,
            message_id=bound_message_id,
            text="",
            command=None,
            is_callback=True,
            callback_id=callback_id,
            callback_data=data,
            sender_id=sender_id,
            callback_message_id=bound_message_id,
        )
    # MaybeInaccessibleMessage or absent message: untrusted without the
    # full token/receipt context, so treat as stale/inert upstream.
    chat_instance = query.get("chat_instance")
    _ = chat_instance
    return TelegramIncoming(
        update_id=update_id,
        chat_id=sender_id,
        message_id=0,
        text="",
        command=None,
        is_callback=True,
        callback_id=callback_id,
        callback_data=data,
        sender_id=sender_id,
        callback_message_id=None,
        callback_inaccessible=True,
    )


def parse_update(raw: Any) -> TelegramIncoming | None:
    """Parse one raw ``getUpdates`` entry into private text/voice, else ``None``."""
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
    if isinstance(text, str) and text:
        command = parse_command(text)
        return TelegramIncoming(
            update_id=update_id,
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            command=command,
        )
    voice_raw = message.get("voice")
    if isinstance(voice_raw, dict):
        file_id = voice_raw.get("file_id")
        duration = voice_raw.get("duration")
        file_size = voice_raw.get("file_size")
        return TelegramIncoming(
            update_id=update_id,
            chat_id=chat_id,
            message_id=message_id,
            text="",
            command=None,
            voice=VoiceAttachment(
                file_id=file_id if isinstance(file_id, str) else "",
                duration_seconds=duration if isinstance(duration, int) else 0,
                file_size_bytes=file_size if isinstance(file_size, int) else None,
            ),
        )
    return None


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
    async def send(self, reply: TelegramReply) -> int | None:
        """Queue or deliver an outbound reply; returns the sent message id when known."""
        raise NotImplementedError

    @abstractmethod
    async def send_voice(self, reply: TelegramVoiceReply) -> int | None:
        """Deliver one OGG/Opus voice reply via ``sendVoice``."""
        raise NotImplementedError

    async def answer_callback(
        self, callback_id: str, text: str | None = None, show_alert: bool = False
    ) -> None:
        """Acknowledge one callback query (neutral UI acknowledgement)."""
        _ = (callback_id, text, show_alert)

    async def edit_reply_markup(
        self, chat_id: int, message_id: int, reply_markup: dict[str, Any] | None
    ) -> None:
        """Edit or remove one inline keyboard without a new chat message."""
        _ = (chat_id, message_id, reply_markup)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Send one ``sendChatAction`` event (typing heartbeat).

        The base implementation is a no-op so offline/test transports stay
        usable; the polling transport overrides it with a real Bot API call.
        Only counts are logged, never message bodies.
        """
        _ = (chat_id, action)

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
        self.sent_voices: list[TelegramVoiceReply] = []
        self.chat_actions: list[tuple[int, str]] = []
        self.callback_answers: list[dict[str, Any]] = []
        self.markup_edits: list[dict[str, Any]] = []
        self._next_message_id = 1000

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send(self, reply: TelegramReply) -> int:
        _check_outbound_envelope(reply.text)
        self._next_message_id += 1
        self.sent.append(reply)
        return self._next_message_id

    async def answer_callback(
        self, callback_id: str, text: str | None = None, show_alert: bool = False
    ) -> None:
        """Record one neutral callback acknowledgement (no business state)."""
        _ = show_alert
        self.callback_answers.append({"id": callback_id, "text": text or ""})

    async def edit_reply_markup(
        self, chat_id: int, message_id: int, reply_markup: dict[str, Any] | None
    ) -> None:
        """Record one keyboard edit/removal (no new chat message)."""
        self.markup_edits.append(
            {"chat_id": chat_id, "message_id": message_id, "markup": reply_markup}
        )

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Record one typing heartbeat event (no network I/O)."""
        self.chat_actions.append((chat_id, action))

    async def send_voice(self, reply: TelegramVoiceReply) -> None:
        if not reply.voice_bytes:
            raise TelegramApiError("telegram voice payload is empty")
        self.sent_voices.append(reply)
        logger.info(
            "telegram voice message sent",
            extra={"chat_id": reply.chat_id, "byte_len": len(reply.voice_bytes)},
        )

    @property
    def running(self) -> bool:
        return self._running


class TelegramApi(ABC):
    """Minimal async Telegram Bot API surface used by the transport."""

    @abstractmethod
    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        """Call a Bot API method and return the ``result`` payload."""
        raise NotImplementedError

    async def download_file(self, file_path: str) -> bytes:
        """Download file content for a ``getFile`` path (voice notes)."""
        raise NotImplementedError

    async def send_voice(self, chat_id: int, ogg_bytes: bytes) -> Any:
        """Send OGG/Opus bytes via ``sendVoice`` (multipart file upload)."""
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

    async def download_file(self, file_path: str) -> bytes:
        """Download ``file_path`` content over file HTTPS (voice notes)."""
        if not file_path or file_path.startswith("/") or ".." in file_path:
            raise TelegramApiError("telegram file path is unsafe")
        return await asyncio.to_thread(self._download_file_sync, file_path)

    def _download_file_sync(self, file_path: str) -> bytes:
        url = f"{self._base_url}/file/bot{self._token}/{file_path}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout_seconds) as resp:
                payload = resp.read()
                return bytes(payload)
        except urllib.error.HTTPError as exc:
            raise _translate_http_error(exc) from exc
        except OSError as exc:
            raise TelegramApiError("telegram file download network error") from exc

    async def send_voice(self, chat_id: int, ogg_bytes: bytes) -> Any:
        """Upload OGG/Opus bytes through ``sendVoice`` multipart."""
        if not ogg_bytes:
            raise TelegramApiError("telegram voice payload is empty")
        return await asyncio.to_thread(self._send_voice_sync, chat_id, bytes(ogg_bytes))

    def _send_voice_sync(self, chat_id: int, ogg_bytes: bytes) -> Any:
        import uuid

        boundary = f"----aa-voice-{uuid.uuid4().hex}"
        url = f"{self._base_url}/bot{self._token}/sendVoice"
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="chat_id"\r\n\r\n')
        body.extend(f"{chat_id}\r\n".encode())
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            b'Content-Disposition: form-data; name="voice"; '
            b'filename="voice.ogg"\r\nContent-Type: audio/ogg\r\n\r\n'
        )
        body.extend(bytes(ogg_bytes))
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        request = urllib.request.Request(
            url,
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise _translate_http_error(exc) from exc
        except OSError as exc:
            raise TelegramApiError("telegram sendVoice network error") from exc
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TelegramApiError("telegram sendVoice bad response") from exc
        return _translate_envelope("sendVoice", envelope)


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
    lowered = description.lower()
    if code == 401 or "unauthorized" in lowered:
        return TelegramAuthError("telegram unauthorized: invalid bot token")
    detail = f": {description}" if description else ""
    if code == 409 or "conflict" in lowered or "another webhook" in lowered:
        return TelegramConflictError(f"telegram polling conflict (409){detail}")
    if code == 429 or "too many requests" in lowered or "rate limit" in lowered:
        return TelegramRateLimitedError(f"telegram rate limited (429){detail}")
    return TelegramApiError(f"telegram http error {code}{detail}")


def _translate_envelope(method: str, envelope: Any) -> Any:
    if not isinstance(envelope, dict):
        raise TelegramApiError(f"telegram {method} bad response")
    if not envelope.get("ok"):
        error_code = envelope.get("error_code")
        description = envelope.get("description", "")
        lowered = description.lower() if isinstance(description, str) else ""
        if error_code == 401 or "unauthorized" in lowered:
            raise TelegramAuthError("telegram unauthorized: invalid bot token")
        if error_code == 409 or "conflict" in lowered:
            raise TelegramConflictError(f"telegram {method} polling conflict (409)")
        if error_code == 429 or "too many requests" in lowered or "rate limit" in lowered:
            raise TelegramRateLimitedError(f"telegram {method} rate limited (429)")
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
        self._sent_voices: list[TelegramVoiceReply] = []
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
    def sent_voices(self) -> list[TelegramVoiceReply]:
        """Voice replies delivered via ``sendVoice`` (for tests)."""
        return list(self._sent_voices)

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

    async def send(self, reply: TelegramReply) -> int | None:
        """Deliver one reply via ``sendMessage`` with bounded retry.

        The #83 transport guard runs first: any ``text`` over the hard
        envelope fails closed here (typed error, no network call, no
        split into multiple messages, response text never logged).
        Returns the Telegram message id when the API provides one.
        """
        _check_outbound_envelope(reply.text)
        payload: dict[str, Any] = {"chat_id": reply.chat_id, "text": reply.text}
        if reply.reply_markup is not None:
            payload["reply_markup"] = dict(reply.reply_markup)
        result = await self._call_with_retry(
            "sendMessage", payload, max_retries=self._max_send_retries
        )
        self._sent.append(reply)
        logger.info(
            "telegram message sent",
            extra={"chat_id": reply.chat_id, "text_len": len(reply.text)},
        )
        if isinstance(result, dict) and isinstance(result.get("message_id"), int):
            return int(result["message_id"])
        return None

    async def send_voice(self, reply: TelegramVoiceReply) -> int | None:
        """Deliver one OGG/Opus voice reply via ``sendVoice``.

        Only sizes are logged, never audio content. Retries are bounded
        like text sends; an empty payload fails closed without network
        I/O so the caller can fall back to text.
        """
        if not reply.voice_bytes:
            raise TelegramApiError("telegram voice payload is empty")
        payload_bytes = bytes(reply.voice_bytes)
        result: Any = None
        attempt = 0
        while True:
            try:
                try:
                    result = await self._send_voice_with_markup(reply, payload_bytes)
                except NotImplementedError as exc:
                    raise TelegramApiError("telegram voice send is not supported") from exc
                break
            except TelegramAuthError:
                raise
            except (TelegramApiError, TimeoutError, OSError) as exc:
                attempt += 1
                if attempt > self._max_send_retries:
                    logger.warning(
                        "telegram voice send failed",
                        extra={"chat_id": reply.chat_id, "attempts": attempt},
                    )
                    raise
                delay = self._backoff_delay(attempt)
                logger.info(
                    "telegram voice send retrying",
                    extra={"chat_id": reply.chat_id, "attempt": attempt, "delay": delay},
                )
                await asyncio.sleep(delay)
                _ = exc
        self._sent_voices.append(reply)
        logger.info(
            "telegram voice message sent",
            extra={"chat_id": reply.chat_id, "byte_len": len(payload_bytes)},
        )
        if isinstance(result, dict) and isinstance(result.get("message_id"), int):
            return int(result["message_id"])
        return None

    async def _send_voice_with_markup(self, reply: TelegramVoiceReply, payload_bytes: bytes) -> Any:
        """Send voice bytes, including inline markup when the API supports it."""
        api = self._api
        send_with_markup = getattr(api, "send_voice_with_markup", None)
        if reply.reply_markup is not None and callable(send_with_markup):
            return await send_with_markup(reply.chat_id, payload_bytes, dict(reply.reply_markup))
        return await api.send_voice(reply.chat_id, payload_bytes)

    async def answer_callback(
        self, callback_id: str, text: str | None = None, show_alert: bool = False
    ) -> None:
        """Send one neutral ``answerCallbackQuery`` acknowledgement."""
        payload: dict[str, Any] = {"callback_query_id": callback_id}
        if text:
            payload["text"] = str(text)[:200]
        if show_alert:
            payload["show_alert"] = True
        try:
            await self._api.call("answerCallbackQuery", payload)
        except TelegramAuthError:
            raise
        except (TelegramApiError, TimeoutError, OSError):
            logger.info("telegram callback acknowledgement failed")

    async def edit_reply_markup(
        self, chat_id: int, message_id: int, reply_markup: dict[str, Any] | None
    ) -> None:
        """Edit or remove one inline keyboard without a new chat message."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
        }
        if reply_markup is None:
            payload["reply_markup"] = {"inline_keyboard": []}
        else:
            payload["reply_markup"] = dict(reply_markup)
        await self._call_with_retry(
            "editMessageReplyMarkup", payload, max_retries=self._max_send_retries
        )

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Send one ``sendChatAction`` typing event (heartbeat, no retry).

        Heartbeat delivery is best-effort: a transient failure is logged
        by category only and never fails the turn. Only counts are
        logged, never message bodies.
        """
        payload: dict[str, Any] = {"chat_id": chat_id, "action": action}
        try:
            await self._api.call("sendChatAction", payload)
        except TelegramAuthError:
            raise
        except (TelegramApiError, TimeoutError, OSError):
            logger.info("telegram chat action failed")
        self._chat_action_count = getattr(self, "_chat_action_count", 0) + 1

    async def fetch_voice_bytes(self, file_id: str) -> bytes:
        """Download one voice file through the existing transport.

        Resolves ``file_id`` via ``getFile`` then fetches the file
        content. Only sizes are logged, never file identifiers or paths.
        """
        if not file_id:
            raise TelegramApiError("telegram voice file id is missing")
        result = await self._call_with_retry(
            "getFile", {"file_id": file_id}, max_retries=self._max_send_retries
        )
        if not isinstance(result, dict):
            raise TelegramApiError("telegram getFile returned an unexpected result")
        file_path = result.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise TelegramApiError("telegram getFile returned no file path")
        data = await self._api.download_file(file_path)
        logger.info("telegram voice file fetched", extra={"byte_len": len(data)})
        return data

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
                if not batch:
                    # Yield when long-poll returns immediately with no
                    # updates (tests use timeout 0 with an instant mock).
                    # Without this, a mock ``getUpdates`` that never blocks
                    # would busy-spin without yielding and starve ordinary
                    # turns/heartbeats sharing the same event loop.
                    # Production long-poll blocks server-side, so this is a
                    # no-op there.
                    await asyncio.sleep(0)
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
            "allowed_updates": ["message", "callback_query"],
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
            "allowed_updates": ["message", "callback_query"],
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

        callback = parse_callback_query(raw)
        if callback is not None:
            # Fast neutral acknowledgement at ingress: stop the client
            # spinner without claiming action success. Business state
            # mutates only later inside the per-chat FIFO worker.
            try:
                await self.answer_callback(str(callback.callback_id or ""))
            except TelegramAuthError:
                raise
            except Exception:
                pass
            logger.info(
                "telegram callback received",
                extra={
                    "update_id": callback.update_id,
                    "chat_id": callback.chat_id,
                    "message_id": callback.message_id,
                },
            )
            if not await self._dispatch(callback):
                return False
            self._received.append(callback)
            self._commit_update_id(update_id)
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
