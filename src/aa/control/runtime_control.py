"""Runtime control boundary.

Owns the requested bot session duration and computes shutdown deadlines so
the asyncio worker can run in the same GitHub Actions job as OpenCode with
a bounded lifetime.
"""

from __future__ import annotations

import time

from aa.control.campaign import RUNTIME_SECONDS


class RuntimeController:
    """Tracks session deadlines and stop requests."""

    def __init__(self, session_duration_seconds: float = 0.0) -> None:
        if session_duration_seconds < 0:
            raise ValueError("session_duration_seconds must be >= 0")
        if session_duration_seconds > float(RUNTIME_SECONDS):
            raise ValueError(
                f"session_duration_seconds must be <= {RUNTIME_SECONDS} (5h campaign max)"
            )
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
