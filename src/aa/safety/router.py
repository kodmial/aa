"""Safety routing boundary.

Classifies inbound text as allowed or blocked before any OpenCode work is
scheduled. The foundation implementation is intentionally conservative and
offline: empty messages are blocked, everything else is allowed. Richer
policy checks belong to later issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SafetyDecision(Enum):
    """Outcome of a safety check."""

    ALLOW = "allow"
    BLOCK = "block"


@dataclass(frozen=True)
class SafetyResult:
    """Result of routing one message through the safety router."""

    decision: SafetyDecision
    reason: str = ""


class SafetyRouter:
    """Routes messages to allow/block outcomes."""

    def __init__(self) -> None:
        self._running = False

    async def start(self) -> None:
        """Enable the router."""
        self._running = True

    async def stop(self) -> None:
        """Disable the router."""
        self._running = False

    @property
    def running(self) -> bool:
        """Whether the router is running."""
        return self._running

    def check(self, text: str) -> SafetyResult:
        """Check ``text`` and return a routing decision."""
        if not text or not text.strip():
            return SafetyResult(decision=SafetyDecision.BLOCK, reason="empty-message")
        return SafetyResult(decision=SafetyDecision.ALLOW, reason="default-allow")
