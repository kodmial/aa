"""Async application lifecycle for the AA Telegram worker."""

from __future__ import annotations

import asyncio
import logging

from aa.config import Settings
from aa.control.runtime_control import RuntimeController
from aa.corpus.budget import MIN_EFFECTIVE_CONTEXT_TOKENS
from aa.corpus.context import CorpusContext
from aa.opencode.runtime import OpenCodeConfig, OpenCodeRuntime, StubOpenCodeRuntime
from aa.safety.router import SafetyRouter
from aa.sessions.coordinator import SessionCoordinator
from aa.telegram.transport import StubTelegramTransport, TelegramTransport

logger = logging.getLogger("aa.app")


class Application:
    """Composes module boundaries and owns startup/shutdown order."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: TelegramTransport | None = None,
        opencode_runtime: OpenCodeRuntime | None = None,
        corpus: CorpusContext | None = None,
        sessions: SessionCoordinator | None = None,
        safety: SafetyRouter | None = None,
        controller: RuntimeController | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport or StubTelegramTransport()
        self.opencode_runtime = opencode_runtime or StubOpenCodeRuntime(
            OpenCodeConfig(
                base_url=settings.opencode_base_url,
                command=settings.opencode_command,
                workdir=settings.opencode_workdir,
                model=settings.opencode_model,
                context_limit_tokens=settings.opencode_context_limit_tokens,
                max_output_tokens=settings.opencode_max_output_tokens,
            )
        )
        self.corpus = corpus or CorpusContext(
            path=settings.aa_corpus_path,
            version=settings.aa_corpus_version,
            effective_context_tokens=(
                settings.opencode_context_limit_tokens
                if settings.opencode_context_limit_tokens > 0
                else MIN_EFFECTIVE_CONTEXT_TOKENS
            ),
        )
        self.sessions = sessions or SessionCoordinator()
        self.safety = safety or SafetyRouter()
        self.controller = controller or RuntimeController(
            session_duration_seconds=settings.bot_session_duration_seconds
        )
        self._running = False

    @property
    def running(self) -> bool:
        """Whether the application is running."""
        return self._running

    async def start(self) -> None:
        """Start all components in dependency order."""
        if self._running:
            return
        self.settings.validate(require_bot_token=False)
        logger.info("starting worker", extra={"config": self.settings.to_safe_dict()})
        await self.controller.start()
        await self.corpus.load()
        await self.opencode_runtime.start()
        await self.sessions.start()
        await self.safety.start()
        await self.transport.start()
        self._running = True
        logger.info("worker started")

    async def stop(self) -> None:
        """Stop all components in reverse order (idempotent)."""
        if not self._running:
            # Still ensure subcomponents are stopped for partial startups.
            await self.transport.stop()
            await self.safety.stop()
            await self.sessions.stop()
            await self.opencode_runtime.stop()
            await self.corpus.unload()
            await self.controller.stop()
            return
        logger.info("stopping worker")
        self._running = False
        await self.transport.stop()
        await self.safety.stop()
        await self.sessions.stop()
        await self.opencode_runtime.stop()
        await self.corpus.unload()
        await self.controller.stop()
        logger.info("worker stopped")

    async def run(self) -> None:
        """Run until the controller requests shutdown."""
        await self.start()
        try:
            while not self.controller.should_stop():
                await asyncio.sleep(0.05)
        finally:
            await self.stop()

    async def __aenter__(self) -> Application:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()


def create_application(settings: Settings) -> Application:
    """Factory used by the entrypoint and tests."""
    return Application(settings)
