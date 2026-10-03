"""Safety routing boundary.

Classifies inbound text before any OpenCode work is scheduled. The router
is deterministic and offline:

- empty messages are blocked;
- acute medical/emergency messages take the emergency route
  (``SafetyDecision.EMERGENCY``) with a bounded safe reply;
- everything else is allowed onto the normal path.

The routing API is small and transport-independent so Telegram integration
can call :meth:`SafetyRouter.check` (or :meth:`SafetyRouter.route`) before
OpenCode. Privacy-safe by contract: logs carry decisions, categories and
lengths only, never raw user message bodies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from aa.safety.emergency import EmergencyCategory, EmergencyClassification, classify_emergency
from aa.safety.response import build_emergency_response

logger = logging.getLogger("aa.safety.router")


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
    categories: tuple[EmergencyCategory, ...] = ()
    language: str = "en"
    classification: EmergencyClassification | None = None


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

        Never logs the message body: only the decision, the matched
        categories and the message length are emitted.
        """
        if not text or not text.strip():
            result = SafetyResult(decision=SafetyDecision.BLOCK, reason="empty-message")
            logger.info(
                "safety decision",
                extra={"decision": result.decision.value, "reason": result.reason, "text_len": 0},
            )
            return result
        classification = classify_emergency(text)
        if classification.is_emergency:
            reason = "emergency:" + ",".join(item.value for item in classification.categories)
            result = SafetyResult(
                decision=SafetyDecision.EMERGENCY,
                reason=reason,
                categories=classification.categories,
                language=classification.language,
                classification=classification,
            )
            logger.info(
                "safety decision",
                extra={
                    "decision": result.decision.value,
                    "reason": result.reason,
                    "text_len": len(text),
                    "language": result.language,
                },
            )
            return result
        result = SafetyResult(decision=SafetyDecision.ALLOW, reason="default-allow")
        logger.info(
            "safety decision",
            extra={
                "decision": result.decision.value,
                "reason": result.reason,
                "text_len": len(text),
            },
        )
        return result

    def route(self, text: str) -> tuple[SafetyResult, str | None]:
        """Route ``text`` and return ``(result, emergency_reply)``.

        ``emergency_reply`` is the bounded safe response when the emergency
        route is taken and ``None`` otherwise. Telegram integration must
        call this before scheduling any OpenCode work: a non-``None`` reply
        takes precedence over ordinary AA answering.
        """
        result = self.check(text)
        if result.decision is SafetyDecision.EMERGENCY and result.classification is not None:
            return result, build_emergency_response(result.classification)
        return result, None


__all__ = ["SafetyDecision", "SafetyResult", "SafetyRouter"]
