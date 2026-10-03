"""OpenCode session/runtime integration boundary.

The worker talks to a local OpenCode runtime (endpoint/process) and never
implements its own LLM client. No live OpenCode calls are made here; this
module only defines configuration plumbing and lifecycle interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenCodeConfig:
    """Pointer configuration for the OpenCode runtime."""

    base_url: str
    command: str
    workdir: str
    model: str = ""
    context_limit_tokens: int = 0
    max_output_tokens: int = 0


class OpenCodeRuntime(ABC):
    """Interface for managing OpenCode sessions."""

    @abstractmethod
    async def start(self) -> None:
        """Prepare the runtime (no subprocess/network in the stub)."""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Release runtime resources."""
        raise NotImplementedError

    @property
    @abstractmethod
    def running(self) -> bool:
        """Whether the runtime is currently running."""
        raise NotImplementedError


class StubOpenCodeRuntime(OpenCodeRuntime):
    """Offline stub recording lifecycle without spawning processes."""

    def __init__(self, config: OpenCodeConfig) -> None:
        self.config = config
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running
