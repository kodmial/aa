"""Telegram transport boundary (outbound long polling only)."""

from __future__ import annotations

from aa.telegram.dispatcher import ChatQueueFullError, ChatTurnDispatcher
from aa.telegram.transport import (
    SUPPORTED_COMMANDS,
    PollingTelegramTransport,
    StubTelegramTransport,
    TelegramApi,
    TelegramApiError,
    TelegramAuthError,
    TelegramIncoming,
    TelegramReply,
    TelegramTransport,
    TelegramUpdate,
    TelegramVoiceReply,
    UrllibTelegramApi,
    VoiceAttachment,
    parse_command,
    parse_update,
)

__all__ = [
    "ChatQueueFullError",
    "ChatTurnDispatcher",
    "SUPPORTED_COMMANDS",
    "PollingTelegramTransport",
    "StubTelegramTransport",
    "TelegramApi",
    "TelegramApiError",
    "TelegramAuthError",
    "TelegramIncoming",
    "TelegramReply",
    "TelegramTransport",
    "TelegramUpdate",
    "TelegramVoiceReply",
    "UrllibTelegramApi",
    "VoiceAttachment",
    "parse_command",
    "parse_update",
]
