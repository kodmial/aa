"""Safety routing boundary.

Classifies inbound text before any OpenCode work is scheduled. Offline and
deterministic:

- empty messages are blocked;
- acute medical/self-harm cases route to ``EMERGENCY`` and receive immediate
  emergency/medical guidance before any ordinary AA-oriented response;
- everything else is allowed.

The router never provides medication dosing or unsupervised detox
instructions; the emergency template is fixed and tested for that property.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aa.safety.emergency import (
    build_emergency_response,
    detect_acute_category,
    detect_language,
    is_emergency,
)


class SafetyDecision(Enum):
    """Outcome of a safety check."""

    ALLOW = "allow"
    BLOCK = "block"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class SafetyResult:
    """Result of routing one message through the safety router."""

    decision: SafetyDecision
    reason: str = ""
    emergency_category: str | None = None
    emergency_response: str = ""


class SafetyRouter:
    """Routes messages to allow/block/emergency outcomes."""

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
        """Check ``text`` and return a routing decision.

        Emergency detection runs before the ordinary allow path so acute
        cases always receive emergency guidance first.
        """
        if not text or not text.strip():
            return SafetyResult(decision=SafetyDecision.BLOCK, reason="empty-message")
        category = detect_acute_category(text)
        if category is not None:
            lang = detect_language(text)
            return SafetyResult(
                decision=SafetyDecision.EMERGENCY,
                reason=f"emergency:{category}",
                emergency_category=category,
                emergency_response=build_emergency_response(lang=lang),
            )
        return SafetyResult(decision=SafetyDecision.ALLOW, reason="default-allow")


__all__ = [
    "SafetyDecision",
    "SafetyResult",
    "SafetyRouter",
    "is_emergency",
    "detect_acute_category",
    "build_emergency_response",
]
