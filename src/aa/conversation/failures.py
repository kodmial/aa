"""Typed unsuccessful turn outcomes and service-error signalling (issue #301).

The generative AA pipeline never returns a canned conversational reply.
Model/provider/retrieval/timeout/verifier failures are typed
unsuccessful outcomes (:class:`TurnFailed`), never synthetic successful
AA conversation. At the transport boundary a minimal outage
notification may be delivered, but it is always a clearly identified
service error (``SERVICE_ERROR_MARKER``), never a fabricated
conversation, and never qualifies as a substantive answer.
"""

from __future__ import annotations

from typing import Any

SERVICE_ERROR_MARKER = "[service-error]"

SERVICE_ERROR_REPLY = (
    f"Сервисная ошибка: временно не могу ответить. Попробуйте позже. {SERVICE_ERROR_MARKER}"
)


class TurnFailed(ValueError):
    """Typed unsuccessful AA turn (never a successful assistant answer)."""

    def __init__(
        self, category: str, detail: str = "", *, telemetry: dict[str, Any] | None = None
    ) -> None:
        super().__init__(f"turn failed [{category}]" + (f": {detail}" if detail else ""))
        self.category = category
        self.detail = detail
        self.telemetry: dict[str, Any] = dict(telemetry) if telemetry else {}


def is_service_error(text: str) -> bool:
    """Whether ``text`` is the transport service-error signal (not dialogue)."""
    return SERVICE_ERROR_MARKER in (text or "")


__all__ = [
    "SERVICE_ERROR_MARKER",
    "SERVICE_ERROR_REPLY",
    "TurnFailed",
    "is_service_error",
]
