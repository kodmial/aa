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
    assert redacted["chat_id"] == aa_logging.REDACTED


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


def test_telegram_identifiers_are_redacted_in_nested_json_extras() -> None:
    stream = io.StringIO()
    aa_logging.configure_logging("INFO", stream=stream)
    logger = aa_logging.get_logger("telegram.privacy")
    logger.info(
        "telegram event processed",
        extra={
            "chat_id": 987654321,
            "update_id": 987654322,
            "message_id": 987654323,
            "sender_id": 987654324,
            "callback_id": "private-callback-987654325",
            "details": {
                "USER_ID": 987654326,
                "callbacks": [
                    {"callback_message_id": 987654327, "status": "ok"},
                    {"file_unique_id": "private-file-987654328"},
                ],
            },
            "scenario_index": 4,
            "attempt_state": "in-flight",
        },
    )
    payload = json.loads(stream.getvalue().strip())
    extras = payload["extra"]
    for key in ("chat_id", "update_id", "message_id", "sender_id", "callback_id"):
        assert extras[key] == aa_logging.REDACTED
    assert extras["details"]["USER_ID"] == aa_logging.REDACTED
    assert extras["details"]["callbacks"][0]["callback_message_id"] == aa_logging.REDACTED
    assert extras["details"]["callbacks"][1]["file_unique_id"] == aa_logging.REDACTED
    assert extras["details"]["callbacks"][0]["status"] == "ok"
    assert extras["scenario_index"] == 4
    assert extras["attempt_state"] == "in-flight"
    for identifier in (
        "987654321",
        "987654322",
        "987654323",
        "987654324",
        "private-callback-987654325",
        "987654326",
        "987654327",
        "private-file-987654328",
    ):
        assert identifier not in stream.getvalue()


def test_telegram_identifier_mapping_redaction_preserves_safe_fields() -> None:
    original = {
        "CHAT_ID": 42,
        "nested": {"update_id": 43, "message_id": "44"},
        "items": [{"sender_id": 45}, {"user_id": 46}],
        "latency_ms": 102.5,
        "status": "delivered",
    }
    redacted = aa_logging.redact_mapping(original)
    assert original["CHAT_ID"] == 42  # Never mutate application data.
    assert redacted["CHAT_ID"] == aa_logging.REDACTED
    assert redacted["nested"] == {
        "update_id": aa_logging.REDACTED,
        "message_id": aa_logging.REDACTED,
    }
    assert redacted["items"] == [
        {"sender_id": aa_logging.REDACTED},
        {"user_id": aa_logging.REDACTED},
    ]
    assert redacted["latency_ms"] == 102.5
    assert redacted["status"] == "delivered"
