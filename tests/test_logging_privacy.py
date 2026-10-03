"""Privacy-safe logging tests."""

from __future__ import annotations

import io
import json
import logging

from aa import logging as aa_logging


def test_redact_secret_keeps_only_length_hint() -> None:
    token = "123456:ABCDEF-secret-token"
    redacted = aa_logging.redact_secret(token)
    assert token not in redacted
    assert "123456" not in redacted
    assert str(len(token)) in redacted


def test_redact_string_masks_token_shapes() -> None:
    token = "123456:ABCDEF-secret-token-value"
    message = f"starting bot with token {token} done"
    redacted = aa_logging.redact_string(message)
    assert token not in redacted
    assert "ABCDEF" not in redacted


def test_redact_mapping_masks_sensitive_keys() -> None:
    token = "123456:ABCDEF-secret-token-value"
    data = {"telegram_bot_token": token, "chat_id": 42, "text": f"hi {token}"}
    redacted = aa_logging.redact_mapping(data)
    assert redacted["telegram_bot_token"] != token
    assert token not in str(redacted)
    assert redacted["chat_id"] == 42


def test_json_logs_never_contain_bot_token() -> None:
    stream = io.StringIO()
    token = "123456:ABCDEF-secret-token-value"
    aa_logging.configure_logging("INFO", stream=stream)
    logger = aa_logging.get_logger("privacy-test")
    logger.info("bot update", extra={"telegram_bot_token": token, "chat_id": 7})
    logger.info("raw %s", token)
    output = stream.getvalue()
    assert token not in output
    assert "ABCDEF" not in output
    # Output remains structured JSON per line.
    for line in output.strip().splitlines():
        payload = json.loads(line)
        assert payload["level"] == "INFO"
        assert "message" in payload


def test_privacy_filter_installed_on_aa_logger() -> None:
    stream = io.StringIO()
    aa_logging.configure_logging("INFO", stream=stream)
    root = logging.getLogger("aa")
    assert root.handlers, "aa logger must have a handler"
    assert any(isinstance(f, aa_logging.PrivacyFilter) for h in root.handlers for f in h.filters)
