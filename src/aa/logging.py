"""Privacy-safe structured logging.

Only the standard library is used. Secrets (notably ``TELEGRAM_BOT_TOKEN``)
are never emitted in full: values are redacted by key name and by pattern
(token-shaped substrings, ``Authorization``/``Bot`` headers).
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

REDACTED = "[REDACTED]"

SENSITIVE_KEYS = frozenset(
    {
        "telegram_bot_token",
        "bot_token",
        "token",
        "secret",
        "authorization",
        "api_key",
        "apikey",
        "password",
    }
)

# Telegram bot tokens look like ``123456:ABC-DEF...``. Also redact generic
# long secret-looking substrings only when explicitly attached to a
# sensitive key (handled in ``redact_mapping``); the pattern below targets
# token-shaped values to avoid over-redacting normal chat text.
_TOKEN_PATTERN = re.compile(r"\b\d{4,15}:[A-Za-z0-9_-]{10,}\b")
_BEARER_PATTERN = re.compile(r"(?i)\b(bearer|bot)\s+[A-Za-z0-9_:\-.~+/=]{8,}\b")


def redact_secret(value: str) -> str:
    """Redact a known secret value, keeping a short non-sensitive hint."""
    if not value:
        return ""
    if len(value) <= 4:
        return REDACTED
    return f"{REDACTED}:len={len(value)}"


def redact_string(text: str) -> str:
    """Redact token-shaped substrings inside free text."""
    if not text:
        return text
    redacted = _TOKEN_PATTERN.sub(REDACTED, text)
    return _BEARER_PATTERN.sub(REDACTED, redacted)


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with sensitive values redacted."""
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            if isinstance(value, str):
                redacted[key] = redact_secret(value)
            else:
                redacted[key] = REDACTED
        elif isinstance(value, str):
            redacted[key] = redact_string(value)
        elif isinstance(value, dict):
            redacted[key] = redact_mapping(value)
        elif isinstance(value, (list, tuple)):
            redacted[key] = [
                redact_mapping(v)
                if isinstance(v, dict)
                else redact_string(v)
                if isinstance(v, str)
                else v
                for v in value
            ]
        else:
            redacted[key] = value
    return redacted


class PrivacyFilter(logging.Filter):
    """Logging filter that redacts secrets from record args and messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args and isinstance(record.args, dict):
            record.args = redact_mapping(record.args)
        # Message itself may already be formatted; redact the rendered text
        # at emit time via the formatter as well. Here we redact ``msg`` when
        # it is a plain string template without args.
        if isinstance(record.msg, str) and not record.args:
            record.msg = redact_string(record.msg)
        return True


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_string(record.getMessage()),
        }
        # Attach structured extras without stdlib record internals.
        reserved = {
            "name",
            "msg",
            "args",
            "created",
            "msecs",
            "relativeCreated",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "thread",
            "threadName",
            "process",
            "processName",
            "message",
            "asctime",
            "taskName",
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in reserved}
        if extras:
            payload["extra"] = redact_mapping(extras)
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", stream: io.TextIOBase | None = None) -> logging.Logger:
    """Configure the ``aa`` root logger with privacy-safe JSON output."""
    root = logging.getLogger("aa")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(PrivacyFilter())
    root.addHandler(handler)
    root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under ``aa`` (configured by ``configure_logging``)."""
    return logging.getLogger(f"aa.{name}")
