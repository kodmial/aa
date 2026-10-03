"""OpenCode session/runtime integration boundary.

The worker talks to a local OpenCode runtime (``opencode serve`` over
loopback HTTP) and never implements its own LLM client. Two runtimes exist:

- :class:`LocalOpenCodeRuntime`: production path. Attaches to an already
  healthy ``opencode serve`` instance or spawns one as a supervised child
  process, polls ``GET /global/health`` until ready, and shuts the child
  down gracefully (SIGTERM, then SIGKILL after a deadline). Attached
  runtimes are never killed on shutdown.
- :class:`StubOpenCodeRuntime`: offline in-memory fake implementing the same
  interface for unit tests and wiring without a real server.

Model/provider (Zen) configuration stays on the OpenCode side; the worker
only forwards the opaque ``OPENCODE_MODEL`` pointer, if set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType

from aa.opencode.client import FakeOpenCodeClient, HttpOpenCodeClient, OpenCodeClient
from aa.opencode.errors import (
    OpenCodeNotReadyError,
    OpenCodeStartupError,
    OpenCodeTimeoutError,
)

logger = logging.getLogger("aa.opencode.runtime")


@dataclass(frozen=True)
class OpenCodeConfig:
    """Pointer configuration for the OpenCode runtime."""

    base_url: str
    command: str
    workdir: str
    model: str = ""
    context_limit_tokens: int = 0
    max_output_tokens: int = 0


def split_base_url(base_url: str) -> tuple[str, int]:
    """Split a loopback base URL into ``(host, port)``.

    An explicit non-zero port is required: with ``--port 0`` OpenCode picks
    a random port that the worker could only discover by parsing server
    logs, which is fragile. Callers must configure the concrete port.
    """
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise OpenCodeStartupError("opencode base URL must be http(s)")
    if parsed.hostname is None:
        raise OpenCodeStartupError("opencode base URL must include a host")
    if parsed.port is None or parsed.port <= 0:
        raise OpenCodeStartupError("opencode base URL must include an explicit port")
    return parsed.hostname, parsed.port


def _probe_health(base_url: str, timeout: float) -> tuple[bool, str]:
    """Blocking single health probe; returns ``(healthy, version)``."""
    url = f"{base_url.rstrip('/')}/global/health"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return False, ""
    if not isinstance(payload, dict):
        return False, ""
    version = payload.get("version")
    return payload.get("healthy") is True, version if isinstance(version, str) else ""


class OpenCodeRuntime(ABC):
    """Interface for managing the local OpenCode runtime."""

    @abstractmethod
    async def start(self) -> None:
        """Attach to or start the runtime (must precede any traffic)."""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Release runtime resources (idempotent, graceful)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def running(self) -> bool:
        """Whether the runtime is currently running."""
        raise NotImplementedError

    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether readiness has been verified (gates Telegram traffic)."""
        raise NotImplementedError

    @abstractmethod
    async def ensure_ready(self, timeout: float | None = None) -> None:
        """Verify readiness, raising startup/timeout errors on failure."""
        raise NotImplementedError

    @property
    @abstractmethod
    def client(self) -> OpenCodeClient:
        """The session client bound to this runtime."""
        raise NotImplementedError


class StubOpenCodeRuntime(OpenCodeRuntime):
    """Offline stub recording lifecycle without spawning processes."""

    def __init__(self, config: OpenCodeConfig, client: OpenCodeClient | None = None) -> None:
        self.config = config
        self._client = client or FakeOpenCodeClient()
        self._running = False
        self._ready = False

    async def start(self) -> None:
        self._running = True
        self._ready = True

    async def stop(self) -> None:
        self._running = False
        self._ready = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def ready(self) -> bool:
        return self._ready

    async def ensure_ready(self, timeout: float | None = None) -> None:
        if not self._running or not self._ready:
            raise OpenCodeNotReadyError("stub opencode runtime is not running")

    @property
    def client(self) -> OpenCodeClient:
        return self._client


class LocalOpenCodeRuntime(OpenCodeRuntime):
    """Production runtime: supervises ``opencode serve`` on loopback.

    ``start()`` first probes :attr:`base_url`; when another ``opencode
    serve`` already answers there, the worker attaches to it and never kills
    it. Otherwise -- when ``manage_process`` is true -- the configured
    command is spawned as ``opencode serve --port <port> --hostname <host>``
    and polled until ``GET /global/health`` reports healthy.
    """

    def __init__(
        self,
        config: OpenCodeConfig,
        client: OpenCodeClient | None = None,
        *,
        manage_process: bool = True,
        ready_timeout: float = 15.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        if ready_timeout <= 0:
            raise ValueError("ready_timeout must be > 0")
        if shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be > 0")
        self.config = config
        self._client = client or HttpOpenCodeClient(config.base_url)
        self._manage_process = manage_process
        self._ready_timeout = ready_timeout
        self._shutdown_timeout = shutdown_timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._running = False
        self._ready = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def client(self) -> OpenCodeClient:
        return self._client

    @property
    def owns_process(self) -> bool:
        """Whether this runtime spawned (and must reap) the server."""
        return self._process is not None

    async def _wait_healthy(self, timeout: float) -> tuple[bool, str]:
        """Poll health until healthy or the deadline; returns last state."""
        deadline = time.monotonic() + timeout
        version = ""
        while time.monotonic() < deadline:
            healthy, version = await asyncio.to_thread(_probe_health, self.config.base_url, 1.0)
            if healthy:
                return True, version
            await asyncio.sleep(0.2)
        return False, version

    async def start(self) -> None:
        """Attach to a healthy server or spawn a supervised one."""
        if self._running:
            return
        split_base_url(self.config.base_url)
        healthy, _ = await asyncio.to_thread(_probe_health, self.config.base_url, 1.0)
        if healthy:
            self._running = True
            self._ready = True
            logger.info("attached to running opencode server")
            return
        if not self._manage_process:
            raise OpenCodeStartupError("opencode server is not reachable for attach")
        await self._spawn()
        self._running = True
        try:
            await self.ensure_ready(self._ready_timeout)
        except Exception:
            await self._terminate_process()
            self._running = False
            raise

    async def _spawn(self) -> None:
        binary = shutil.which(self.config.command) if self.config.command else None
        if binary is None:
            raise OpenCodeStartupError(f"opencode command not found: {self.config.command!r}")
        host, port = split_base_url(self.config.base_url)
        args = [binary, "serve", "--port", str(port), "--hostname", host]
        try:
            self._process = subprocess.Popen(
                args,
                cwd=self.config.workdir or ".",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise OpenCodeStartupError("failed to launch opencode serve") from exc
        logger.info("launched opencode serve")

    async def ensure_ready(self, timeout: float | None = None) -> None:
        """Poll health; raise startup/timeout errors before traffic flows."""
        if not self._running and self._process is None:
            raise OpenCodeNotReadyError("opencode runtime has not been started")
        if self._process is not None and self._process.poll() is not None:
            self._ready = False
            raise OpenCodeStartupError("opencode serve exited before becoming ready")
        limit = self._ready_timeout if timeout is None else timeout
        if limit <= 0:
            raise ValueError("timeout must be > 0")
        healthy, _ = await self._wait_healthy(limit)
        if not healthy:
            if self._process is not None and self._process.poll() is not None:
                self._ready = False
                raise OpenCodeStartupError("opencode serve exited before becoming ready")
            raise OpenCodeTimeoutError("opencode server did not become ready in time")
        self._ready = True

    async def stop(self) -> None:
        """Graceful shutdown: SIGTERM, deadline, SIGKILL fallback (idempotent)."""
        self._running = False
        self._ready = False
        await self._terminate_process()
        logger.info("opencode runtime stopped")

    async def _terminate_process(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait), self._shutdown_timeout)
        except TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), self._shutdown_timeout)
            except TimeoutError:
                logger.warning("opencode serve did not exit after SIGKILL")

    async def __aenter__(self) -> LocalOpenCodeRuntime:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()
