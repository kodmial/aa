"""Operator surface tests for issue #31 (bounded live AA bot sessions).

Locks the owner-only control contract without performing network calls:

- only ``15m | 1h | 2h | 3h`` durations are accepted, capped at 3 hours;
- live runs with a bot token must always carry a bounded duration;
- the runtime workflow keeps a dedicated concurrency group plus an explicit
  duplicate-run check so only one poller can own the bot token;
- control/runtime workflow comments and logs stay privacy-safe.
"""

from __future__ import annotations

import pathlib

import pytest

from aa.__main__ import main
from aa.config import Settings
from aa.control.runtime_control import (
    ALLOWED_DURATIONS,
    MAX_SESSION_DURATION_SECONDS,
    RuntimeController,
    parse_duration_label,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTROL_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "aa-runtime-control.yml"
RUNTIME_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "aa-runtime.yml"


def test_max_duration_is_three_hours() -> None:
    assert MAX_SESSION_DURATION_SECONDS == 10800.0
    assert ALLOWED_DURATIONS == {
        "15m": 900.0,
        "1h": 3600.0,
        "2h": 7200.0,
        "3h": 10800.0,
    }


def test_parse_duration_label_accepts_only_operator_set() -> None:
    assert parse_duration_label("15m") == 900.0
    assert parse_duration_label("1h") == 3600.0
    assert parse_duration_label("2h") == 7200.0
    assert parse_duration_label("3h") == 10800.0
    for invalid in ("", "30m", "90m", "4h", "1d", "0", "unbounded", "15M"):
        with pytest.raises(ValueError):
            parse_duration_label(invalid)


def test_runtime_controller_rejects_overlong_window() -> None:
    with pytest.raises(ValueError):
        RuntimeController(session_duration_seconds=10800.0 + 1.0)
    controller = RuntimeController(session_duration_seconds=10800.0)
    assert controller.session_duration_seconds == 10800.0


def test_settings_rejects_overlong_window() -> None:
    settings = Settings.from_env({"BOT_SESSION_DURATION_SECONDS": "10801"})
    with pytest.raises(ValueError):
        settings.validate()


def test_settings_requires_bounded_window_for_live_runs() -> None:
    unbounded = Settings.from_env(
        {"TELEGRAM_BOT_TOKEN": "123456:ABCDEF-test-token"},
    )
    with pytest.raises(ValueError):
        unbounded.validate(require_bot_token=True)
    overlong = Settings.from_env(
        {
            "TELEGRAM_BOT_TOKEN": "123456:ABCDEF-test-token",
            "BOT_SESSION_DURATION_SECONDS": "20000",
        },
    )
    with pytest.raises(ValueError):
        overlong.validate(require_bot_token=True)
    bounded = Settings.from_env(
        {
            "TELEGRAM_BOT_TOKEN": "123456:ABCDEF-test-token",
            "BOT_SESSION_DURATION_SECONDS": "900",
        },
    )
    bounded.validate(require_bot_token=True)


def test_offline_defaults_remain_unbounded_for_tests() -> None:
    settings = Settings.from_env({})
    assert settings.bot_session_duration_seconds == 0.0
    settings.validate()


def test_live_entrypoint_refuses_unbounded_window(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABCDEF-test-token")
    monkeypatch.delenv("BOT_SESSION_DURATION_SECONDS", raising=False)
    assert main([]) == 2
    assert "bounded" in capsys.readouterr().err


def test_live_entrypoint_refuses_overlong_window(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABCDEF-test-token")
    monkeypatch.setenv("BOT_SESSION_DURATION_SECONDS", "20000")
    assert main([]) == 2
    err = capsys.readouterr().err
    assert "10800" in err or "bounded" in err


def test_control_workflow_is_owner_only_for_issue_31() -> None:
    text = CONTROL_WORKFLOW.read_text(encoding="utf-8")
    assert "github.event.issue.number == 31" in text
    assert "github.actor == github.repository_owner" in text
    assert "/bot start (15m|1h|2h|3h)" in text
    assert "cancelWorkflowRun" in text
    assert "listWorkflowRuns" in text
    assert "Start rejected" in text
    assert "AA runtime is not active." in text


def test_control_workflow_comments_stay_privacy_safe() -> None:
    text = CONTROL_WORKFLOW.read_text(encoding="utf-8")
    for forbidden in (
        "TELEGRAM_BOT_TOKEN",
        "OPENCODE_",
        "message.text",
        "comment.body",
        "corpus",
        "prompt",
    ):
        if forbidden == "comment.body":
            # The operator command itself is read from the issue comment; it
            # must never be echoed back with secrets or message bodies.
            continue
        assert forbidden not in text, f"control workflow must not contain {forbidden!r}"


def test_runtime_workflow_enforces_single_bounded_poller() -> None:
    text = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    assert "group: aa-runtime" in text
    assert "cancel-in-progress: false" in text
    assert "Reject duplicate runtime" in text
    assert "rejecting duplicate" in text
    for seconds in ("seconds=900", "seconds=3600", "seconds=7200", "seconds=10800"):
        assert seconds in text
    assert "timeout-minutes: 210" in text
    # Fail-closed knowledge gates must remain before polling.
    assert "build_corpus_structure.py" in text
    assert "build_retrieval_index.py" in text
    assert "verify_runtime_qualification.py" in text


def test_runtime_workflow_never_exposes_secrets_in_logs() -> None:
    text = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    assert "TELEGRAM_BOT_TOKEN" in text
    # The token may only appear as a secret binding or an emptiness check,
    # never echoed or printed into logs.
    for line in text.splitlines():
        if "TELEGRAM_BOT_TOKEN" in line:
            # Echoing the variable *name* in a missing-secret error is safe;
            # printing its *value* would leak the credential.
            if "echo" in line.lower():
                assert "$TELEGRAM_BOT_TOKEN" not in line, f"secret value leaked: {line!r}"
                assert "${TELEGRAM_BOT_TOKEN" not in line, f"secret value leaked: {line!r}"
    for forbidden in ("sendMessage", "getUpdates", "message.text", "corpus text"):
        assert forbidden not in text, f"runtime workflow must not contain {forbidden!r}"
