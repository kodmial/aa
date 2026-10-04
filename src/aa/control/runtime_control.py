"""Runtime control boundary.

Owns the requested bot session duration and computes shutdown deadlines so
the asyncio worker can run in the same GitHub Actions job as OpenCode with
a bounded lifetime.

The operator surface (issue #31) only permits ``15m | 1h | 2h | 3h``. Every
live run is bounded and the maximum duration is 3 hours (10800 seconds).
"""

from __future__ import annotations

import time

MAX_SESSION_DURATION_SECONDS = 10800.0

ALLOWED_DURATIONS: dict[str, float] = {
    "15m": 900.0,
    "1h": 3600.0,
    "2h": 7200.0,
    "3h": 10800.0,
}


def parse_duration_label(label: str) -> float:
    """Parse an operator duration label into seconds.

    Only ``15m | 1h | 2h | 3h`` are valid; anything else raises ``ValueError``.
    """
    key = label.strip()
    try:
        return ALLOWED_DURATIONS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported bot duration: {label!r}") from exc


class RuntimeController:
    """Tracks session deadlines and stop requests."""

    def __init__(self, session_duration_seconds: float = 0.0) -> None:
        if session_duration_seconds < 0:
            raise ValueError("session_duration_seconds must be >= 0")
        if session_duration_seconds > MAX_SESSION_DURATION_SECONDS:
            raise ValueError("session_duration_seconds must be <= 10800 (max 3h)")
        self.session_duration_seconds = session_duration_seconds
        self._deadline: float | None = None
        self._stop_requested = False
        self._running = False

    async def start(self) -> None:
        """Start the controller and arm the session deadline."""
        self._running = True
        self._stop_requested = False
        if self.session_duration_seconds > 0:
            self._deadline = time.monotonic() + self.session_duration_seconds
        else:
            self._deadline = None

    async def stop(self) -> None:
        """Request a shutdown."""
        self._stop_requested = True
        self._running = False

    @property
    def running(self) -> bool:
        """Whether the controller is running."""
        return self._running

    @property
    def stop_requested(self) -> bool:
        """Whether a stop has been requested."""
        return self._stop_requested

    @property
    def deadline(self) -> float | None:
        """Monotonic deadline for the session, if bounded."""
        return self._deadline

    def time_remaining(self) -> float | None:
        """Seconds remaining before the deadline, if bounded."""
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    def should_stop(self) -> bool:
        """Whether the worker should shut down now."""
        if self._stop_requested:
            return True
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return True
        return False
