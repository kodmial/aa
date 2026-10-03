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
    """Pointer configuration for the OpenCode runtime.

    The canonical AA corpus must be kept in the stable prompt/session prefix
    (or equivalent cached context) with an effective model context of at
    least ``MIN_EFFECTIVE_CONTEXT_TOKENS``. ``context_limit_tokens == 0``
    means "unspecified" for local boot; production must configure >= 200k.
    """

    base_url: str
    command: str
    workdir: str
    model: str = ""
    context_limit_tokens: int = 0
    max_output_tokens: int = 0

    def effective_context_tokens(self) -> int:
        """Return the configured limit, defaulting to the 200k minimum."""
        from aa.corpus.budget import MIN_EFFECTIVE_CONTEXT_TOKENS

        if self.context_limit_tokens > 0:
            return self.context_limit_tokens
        return MIN_EFFECTIVE_CONTEXT_TOKENS

    def validate_context_contract(self) -> None:
        """Fail closed when an explicit limit is below the 200k minimum."""
        from aa.corpus.budget import MIN_EFFECTIVE_CONTEXT_TOKENS

        if 0 < self.context_limit_tokens < MIN_EFFECTIVE_CONTEXT_TOKENS:
            raise ValueError(
                f"OPENCODE_CONTEXT_LIMIT_TOKENS={self.context_limit_tokens} "
                f"is below the required minimum {MIN_EFFECTIVE_CONTEXT_TOKENS}; "
                "the full canonical corpus must fit without truncation"
            )


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
