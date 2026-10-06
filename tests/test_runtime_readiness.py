"""Regression tests for authoritative runtime readiness (issue #144).

Proves the operator-facing control-plane defect stays fixed: the live
runtime publishes a privacy-safe READY marker to control issue #31 only
after long polling is live, `/bot status` distinguishes starting from
ready, startup failures publish STARTUP_FAILED (never an ambiguous
in_progress), and a GitHub Actions step that is merely in_progress is
never treated as readiness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aa.control.campaign import usable_poller_active
from aa.control.readiness import (
    RuntimeIdentity,
    assert_marker_privacy_safe,
    find_ready_for_run,
    format_ready_marker,
    format_startup_failed_marker,
    has_ready_for_run,
    is_usable_poller,
    parse_ready_marker,
    parse_startup_failed_marker,
    resolve_poller_state,
    resolve_runtime_identity,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SHA = "a" * 40


def _read(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_ready_marker_round_trip_is_privacy_safe() -> None:
    marker = format_ready_marker(run_id=111, sha=SHA, ready_at=1700000000, ordinal=2)
    assert "aa-runtime-ready" in marker
    assert "111" in marker
    assert SHA in marker
    parsed = parse_ready_marker(f"prefix\n{marker}\nsuffix")
    assert parsed is not None
    assert parsed.run_id == 111
    assert parsed.sha == SHA
    assert parsed.ordinal == 2
    assert_marker_privacy_safe(marker)
    assert parse_ready_marker("no marker") is None
    with pytest.raises(ValueError):
        format_ready_marker(run_id=111, sha="short", ready_at=1, ordinal=1)
    with pytest.raises(ValueError):
        format_ready_marker(run_id=111, sha=SHA, ready_at=1, ordinal=9)


def test_startup_failed_marker_round_trip_with_bounded_category() -> None:
    marker = format_startup_failed_marker(
        run_id=112, sha=SHA, failed_at=1700000001, ordinal=1, category="telegram-auth"
    )
    parsed = parse_startup_failed_marker(marker)
    assert parsed is not None
    assert parsed.run_id == 112
    assert parsed.category == "telegram-auth"
    assert_marker_privacy_safe(marker)
    with pytest.raises(ValueError):
        format_startup_failed_marker(
            run_id=112, sha=SHA, failed_at=1, ordinal=1, category="not-a-category"
        )


def test_in_progress_alone_is_starting_never_ready() -> None:
    # The core regression: a GitHub Actions step that is merely in_progress
    # (run_active True, no READY marker) is starting, not ready.
    assert (
        resolve_poller_state(
            run_active=True, run_conclusion=None, has_ready=False, has_failed=False
        )
        == "starting"
    )
    assert is_usable_poller(run_active=True, has_ready=False) is False
    assert usable_poller_active(runtime_active=True, has_ready=False) is False
    # Only a trusted READY marker for the exact active run is usable.
    assert (
        resolve_poller_state(run_active=True, run_conclusion=None, has_ready=True, has_failed=False)
        == "ready"
    )
    assert is_usable_poller(run_active=True, has_ready=True) is True
    assert usable_poller_active(runtime_active=True, has_ready=True) is True
    # Concluded runs never report ready even with a stale marker present.
    assert (
        resolve_poller_state(
            run_active=False, run_conclusion="success", has_ready=True, has_failed=False
        )
        == "completed"
    )
    assert (
        resolve_poller_state(
            run_active=False, run_conclusion="failure", has_ready=False, has_failed=True
        )
        == "failed"
    )


def test_ready_lookup_is_per_exact_run() -> None:
    first = format_ready_marker(run_id=201, sha=SHA, ready_at=1700000000, ordinal=1)
    bodies = ["hello", first]
    assert has_ready_for_run(bodies, 201) is True
    assert has_ready_for_run(bodies, 202) is False
    assert find_ready_for_run(bodies, 202) is None
    assert find_ready_for_run(bodies, 201) is not None


def test_runtime_identity_resolves_offline_to_none() -> None:
    assert (
        resolve_runtime_identity(
            environ={"GITHUB_RUN_ID": "", "GITHUB_SHA": "", "AA_CAMPAIGN_SEQ": "1"}
        )
        is None
    )
    identity = resolve_runtime_identity(
        environ={"GITHUB_RUN_ID": "777", "GITHUB_SHA": SHA, "AA_CAMPAIGN_SEQ": "3"}
    )
    assert identity is not None
    assert identity.run_id == 777
    assert identity.ordinal == 3


async def test_ready_emitted_only_after_polling_live() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime

    published: list[str] = []

    async def _sink(marker: str) -> None:
        published.append(marker)

    async def _delegate(thread: str, text: str) -> str:
        return "Понял вас. Давайте разберём спокойно."

    app = Application(
        Settings.from_env(),
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=GraphTurnRuntime(delegate=_delegate),
        readiness_publisher=_sink,
        runtime_identity=RuntimeIdentity(run_id=301, sha=SHA, ordinal=1),
    )
    assert app.readiness_marker is None
    assert app._transport_polling_live() is False
    await app.start()
    try:
        assert app._transport_polling_live() is True
        assert app.readiness_marker is not None
        assert published == [app.readiness_marker]
        parsed = parse_ready_marker(app.readiness_marker)
        assert parsed is not None and parsed.run_id == 301
    finally:
        await app.stop()


async def test_startup_failure_publishes_failed_not_ready() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.control.readiness import RuntimeIdentity
    from aa.opencode.errors import OpenCodeNotReadyError
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime

    published: list[str] = []

    async def _sink(marker: str) -> None:
        published.append(marker)

    class _NeverReady(StubOpenCodeRuntime):
        async def ensure_ready(self, timeout: float | None = None) -> None:
            raise OpenCodeNotReadyError("not ready")

    app = Application(
        Settings.from_env(),
        opencode_runtime=_NeverReady(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        readiness_publisher=_sink,
        runtime_identity=RuntimeIdentity(run_id=302, sha=SHA, ordinal=2),
    )
    with pytest.raises(OpenCodeNotReadyError):
        await app.start()
    assert app.readiness_marker is None
    assert app.startup_failure_marker is not None
    assert published == [app.startup_failure_marker]
    parsed = parse_startup_failed_marker(app.startup_failure_marker)
    assert parsed is not None
    assert parsed.category == "opencode-not-ready"
    # Bounded recovery: the controller is stopped, never left ambiguous.
    assert app.controller.should_stop() is True


async def test_telegram_auth_failure_category() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.control.readiness import RuntimeIdentity
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.telegram.transport import TelegramAuthError

    published: list[str] = []

    async def _sink(marker: str) -> None:
        published.append(marker)

    class _BadToken(StubOpenCodeRuntime):
        pass

    from aa.telegram.transport import StubTelegramTransport

    class _AuthFail(StubTelegramTransport):
        async def start(self) -> None:
            raise TelegramAuthError("telegram unauthorized: invalid bot token")

    app = Application(
        Settings.from_env(),
        transport=_AuthFail(),
        opencode_runtime=_BadToken(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        readiness_publisher=_sink,
        runtime_identity=RuntimeIdentity(run_id=303, sha=SHA, ordinal=1),
    )
    with pytest.raises(TelegramAuthError):
        await app.start()
    assert app.readiness_marker is None
    parsed = parse_startup_failed_marker(app.startup_failure_marker or "")
    assert parsed is not None
    assert parsed.category == "telegram-auth"


def test_control_workflow_reports_starting_ready_failed_states() -> None:
    text = _read("aa-runtime-control.yml")
    assert "aa-runtime-ready" in text
    assert "aa-runtime-startup-failed" in text
    assert "starting" in text
    assert "ready" in text
    assert "pollerStateForRun" in text or "poller_state" in text.lower()
    # Duplicate prevention is unchanged.
    assert "never queue a second poller" in text
    assert "cancel-in-progress: false" in text


def test_reconciler_treats_only_ready_as_usable_poller() -> None:
    text = _read("aa-runtime-reconciler.yml")
    assert "aa-runtime-ready" in text
    assert "never queue a second poller" in text
    assert "starting" in text
    assert "ready poller" in text.lower() or "ready" in text


def test_runtime_workflow_plumbs_campaign_ordinal() -> None:
    text = _read("aa-runtime.yml")
    assert "AA_CAMPAIGN_SEQ" in text
    assert "AA_ENABLE_READY_PUBLISH" in text
    assert "aa-runtime-ready" in text or "READY marker" in text
