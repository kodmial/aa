"""P0 kodmial/aa#547: provider 429 aborts a live scenario promptly.

Proven failure on exact main ``d23287a758d2775749823981cc51a9e5545bf24d``
(run 44611952510, qualification #7): Gate C scenario 4
(``family=followup-ellipsis``) received an ``OpenCodeRateLimitError`` from a
verifier ``message-structured`` call. The application recorded the fatal
worker-lifecycle failure, but the per-chat dispatcher swallowed the re-raised
exception and the live lane only watched for a delivered reply. The scenario
therefore waited out the full 1800-second inactivity watchdog and was
misclassified as ``stuck-provider`` instead of retiring the runner for
fresh-VM checkpoint recovery.

The lane now observes the application's recorded fatal error in every live
delivery wait and re-raises it, so the existing 429 runner-retire contract
(exit 75 / ``OPENCODE_429_RESTART_REQUIRED``) is preserved.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from aa.app import Application
from aa.config import Settings
from aa.opencode.errors import OpenCodeRateLimitError
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.telegram.transport import TelegramIncoming


def _app() -> Application:
    return Application(
        Settings.from_env({}),
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
    )


async def test_rate_limit_turn_records_application_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 turn must surface the lifecycle signal the run loop re-raises."""
    app = _app()
    error = OpenCodeRateLimitError("opencode request rate-limited: http=429")

    async def _boom(incoming: TelegramIncoming) -> None:
        _ = incoming
        raise error

    monkeypatch.setattr(app, "_handle_telegram_update", _boom)
    incoming = TelegramIncoming(
        update_id=1,
        chat_id=920002,
        message_id=1,
        text="synthetic scenario text",
    )
    with pytest.raises(OpenCodeRateLimitError) as raised:
        await app._process_dispatched_update(incoming)
    assert raised.value is error
    assert app.fatal_error is error
    assert app.controller.should_stop() is True


def test_live_lane_reraises_recorded_fatal_error() -> None:
    from aa.qualification.product_contract_live import _raise_if_runtime_fatal

    error = OpenCodeRateLimitError("opencode request rate-limited: http=429")
    with pytest.raises(OpenCodeRateLimitError) as raised:
        _raise_if_runtime_fatal(SimpleNamespace(fatal_error=error))
    assert raised.value is error


def test_live_lane_fatal_check_is_a_noop_without_failure() -> None:
    from aa.qualification.product_contract_live import _raise_if_runtime_fatal

    _raise_if_runtime_fatal(SimpleNamespace(fatal_error=None))
    _raise_if_runtime_fatal(SimpleNamespace())


def test_live_delivery_waits_observe_fatal_error_before_inactivity() -> None:
    """Every live delivery wait must check the fatal signal before sleeping."""
    from aa.qualification.product_contract_live import run_live_telegram_evidence_lane

    source = inspect.getsource(run_live_telegram_evidence_lane)
    assert source.count("_raise_if_runtime_fatal(app)") == 3
    for marker in (
        'stop_reason = ""\n                while len(api.sent_texts) <= before:',
        'raw_stop_reason = ""\n                while len(api.sent_texts) <= before:',
        "while api.sent_voices <= before_voice and len(api.sent_texts) <= before_text:",
    ):
        start = source.index(marker)
        fatal_index = source.index("_raise_if_runtime_fatal(app)", start)
        sleep_index = source.index("await asyncio.sleep(0.02)", start)
        assert fatal_index < sleep_index
