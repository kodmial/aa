"""Async lifecycle tests."""

from __future__ import annotations

import asyncio

from aa.app import Application, create_application
from aa.config import Settings
from aa.opencode.runtime import LocalOpenCodeRuntime, OpenCodeConfig, StubOpenCodeRuntime


def _settings(**overrides: object) -> Settings:
    base: dict[str, str] = {}
    return Settings.from_env(base)


def _stub_runtime() -> StubOpenCodeRuntime:
    return StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )


async def test_lifecycle_starts_and_shuts_down_cleanly() -> None:
    app = Application(_settings(), opencode_runtime=_stub_runtime())
    assert not app.running
    await app.start()
    assert app.running
    assert app.transport.running
    assert app.opencode_runtime.running
    assert app.opencode_runtime.ready
    assert app.sessions.running
    assert app.safety.running
    assert app.controller.running
    assert app.corpus.loaded
    await app.stop()
    assert not app.running
    assert not app.transport.running
    assert not app.opencode_runtime.running
    assert not app.corpus.loaded


async def test_stop_is_idempotent() -> None:
    app = Application(_settings(), opencode_runtime=_stub_runtime())
    await app.start()
    await app.stop()
    await app.stop()
    assert not app.running


async def test_session_window_starts_after_runtime_readiness() -> None:
    class SlowReadyRuntime(StubOpenCodeRuntime):
        async def ensure_ready(self, timeout: float | None = None) -> None:
            await asyncio.sleep(0.08)
            await super().ensure_ready(timeout)

    settings = Settings.from_env({"BOT_SESSION_DURATION_SECONDS": "0.2"})
    runtime = SlowReadyRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )
    app = Application(settings, opencode_runtime=runtime)
    await app.start()
    try:
        remaining = app.controller.time_remaining()
        assert remaining is not None
        assert remaining > 0.15
    finally:
        await app.stop()


async def test_run_respects_session_duration() -> None:
    settings = Settings.from_env({"BOT_SESSION_DURATION_SECONDS": "0.05"})
    app = Application(settings, opencode_runtime=_stub_runtime())
    await asyncio.wait_for(app.run(), timeout=5.0)
    assert not app.running


async def test_context_manager_lifecycle() -> None:
    app = Application(_settings(), opencode_runtime=_stub_runtime())
    async with app as started:
        assert started.running
    assert not app.running


def test_check_boot_path() -> None:
    from aa.__main__ import main

    assert main(["--check"]) == 0


def test_default_runtime_is_local_opencode() -> None:
    app = create_application(_settings())
    assert isinstance(app.opencode_runtime, LocalOpenCodeRuntime)
