"""In-flight attempt ledger for local OpenCode HTTP operations (kodmial/aa#431).

The previous request-latency audit recorded one entry *after* each local
HTTP call completed. A request that never completed therefore published
nothing, and a stalled turn looked identical to an unobservable one. It
also made :func:`aa.qualification.live_turn_repair.request_state_from_audit`
misread a *previous* completed attempt as proof that the *current* request
finished. This module fixes both problems at the source:

- ``register_start`` publishes an ``attempt-start`` record *before* the
  blocking ``urllib`` worker thread is spawned, carrying the turn-causality
  id, pipeline stage, logical agent identity, requested primary/fallback
  models and a monotonic start timestamp. The record stays ``in-flight``
  until an honest terminal state is written.
- ``finish`` writes exactly one terminal state per attempt
  (``completed|failed|cancelled|aborted``). A late outcome (a cancelled
  coroutine whose worker thread eventually completes) is recorded as
  ``late_outcome`` evidence on the same attempt, never as progress for a
  different turn.
- ``mark_worker_done`` independently corroborates the worker thread's
  fate. ``Task.done()`` proves nothing about the underlying thread, so a
  ``cancelled`` attempt is only *proven quiesced* once its worker future
  terminally completed (cleanly before the thread started, or later with
  a late outcome). Qualification quiescence gates consume
  :meth:`proven_quiesced` and never a bare task flag.
- The ledger is a plain data structure owned by the loop-side client and
  read by the independent snapshot observer, so evidence survives even
  when ``urllib.read()`` is stuck inside a worker thread.

Every entry is privacy-safe: operation category plus bounded truncations;
never prompt text, replies, session ids, credentials, or raw provider
error bodies.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

IN_FLIGHT_STATE = "in-flight"
TERMINAL_ATTEMPT_STATES: tuple[str, ...] = ("completed", "failed", "cancelled", "aborted")
LATE_OUTCOMES: tuple[str, ...] = TERMINAL_ATTEMPT_STATES
# A terminal state corroborated by the worker future's own outcome.
QUIESCENCE_STATES: frozenset[str] = frozenset({"completed", "failed", "aborted"})

_MAX_ATTEMPT_RECORDS = 2048
_MAX_CAUSALITY_LEN = 64
_MAX_FIELD_LEN = 64
_MAX_AGENT_LEN = 128
_MAX_MODEL_LEN = 128


def _truncate(value: object, limit: int) -> str:
    return str(value or "")[:limit]


class AttemptLedger:
    """Bounded, append-and-transition ledger of OpenCode HTTP attempts.

    One attempt record is created per HTTP operation. ``state`` starts at
    ``in-flight`` and transitions at most once to a terminal value; later
    transitions are ignored so a stale worker callback can never overwrite
    the honest terminal verdict already recorded.
    """

    def __init__(self, *, max_records: int = _MAX_ATTEMPT_RECORDS) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be > 0")
        self._entries: list[dict[str, Any]] = []
        self._max_records = max_records
        self._lock = threading.Lock()

    def register_start(
        self,
        *,
        operation: str,
        causality_id: str = "",
        stage: str = "",
        agent: str = "",
        audit_agent: str = "",
        model: str = "",
        fallback_model: str = "",
        started_monotonic: float | None = None,
    ) -> str:
        """Publish an ``attempt-start`` record and return its unique id."""
        attempt_id = uuid.uuid4().hex
        with self._lock:
            self._entries.append(
                {
                    "attempt_id": attempt_id,
                    "event": "attempt-start",
                    "operation": _truncate(operation, _MAX_FIELD_LEN),
                    "state": IN_FLIGHT_STATE,
                    "attempt_started_monotonic": (
                        float(started_monotonic)
                        if started_monotonic is not None
                        else time.monotonic()
                    ),
                    "causality_id": _truncate(causality_id, _MAX_CAUSALITY_LEN),
                    "stage": _truncate(stage, _MAX_FIELD_LEN),
                    "agent": _truncate(agent, _MAX_AGENT_LEN),
                    "audit_agent": _truncate(audit_agent, _MAX_AGENT_LEN),
                    "model": _truncate(model, _MAX_MODEL_LEN),
                    "fallback_model": _truncate(fallback_model, _MAX_MODEL_LEN),
                }
            )
            if len(self._entries) > self._max_records:
                del self._entries[: len(self._entries) - self._max_records]
        return attempt_id

    def _find(self, attempt_id: str) -> dict[str, Any] | None:
        for entry in reversed(self._entries):
            candidate: dict[str, Any] = entry
            if candidate.get("attempt_id") == attempt_id:
                return candidate
        return None

    @staticmethod
    def _write_late_outcome_unlocked(
        entry: dict[str, Any],
        state: str,
        latency_ms: float | None,
        failure_category: str,
    ) -> bool:
        """Write one late outcome idempotently (caller holds ``_lock``).

        The worker side and the loop side can both reach this write. The
        first terminal verdict wins, and both paths use the exact same
        field order, so the recorded evidence never depends on which path
        wins the race.
        """
        if entry.get("late_outcome") in LATE_OUTCOMES:
            return False
        entry["late_outcome"] = state
        if latency_ms is not None:
            entry["late_latency_ms"] = round(max(0.0, float(latency_ms)), 1)
        if failure_category:
            entry["late_failure_category"] = _truncate(failure_category, _MAX_FIELD_LEN)
        return True

    def finish(
        self,
        attempt_id: str,
        *,
        state: str,
        latency_ms: float | None = None,
        failure_category: str = "",
        late: bool = False,
    ) -> bool:
        """Transition one attempt past ``in-flight`` exactly once.

        ``late=True`` records the ``late_outcome`` of an attempt whose
        awaiting coroutine was already cancelled while the worker thread
        kept running. It never flips the attempt's authoritative ``state``.
        """
        normalized = _truncate(state, _MAX_FIELD_LEN)
        if normalized not in LATE_OUTCOMES:
            raise ValueError(f"unknown terminal attempt state: {state}")
        with self._lock:
            entry = self._find(attempt_id)
            if entry is None:
                return False
            if late:
                return self._write_late_outcome_unlocked(
                    entry, normalized, latency_ms, failure_category
                )
            if entry.get("state") != IN_FLIGHT_STATE:
                # A terminal state already exists; the recorded state stays the
                # authoritative newest verdict.
                return False
            entry["state"] = normalized
            entry["attempt_ended_monotonic"] = time.monotonic()
            if latency_ms is not None:
                entry["latency_ms"] = round(max(0.0, float(latency_ms)), 1)
            if failure_category:
                entry["failure_category"] = _truncate(failure_category, _MAX_FIELD_LEN)
            return True

    def record_late_outcome_if_cancelled(
        self,
        attempt_id: str,
        state: str,
        latency_ms: float | None = None,
        failure_category: str = "",
    ) -> bool:
        """Atomically record a late outcome only for a cancelled attempt.

        The worker thread's fate is recorded independently of the awaiting
        coroutine. This single locked operation closes the check-then-act
        window between observing ``cancelled`` and writing ``late_outcome``.
        """
        normalized = _truncate(state, _MAX_FIELD_LEN)
        if normalized not in LATE_OUTCOMES:
            raise ValueError(f"unknown terminal attempt state: {state}")
        with self._lock:
            entry = self._find(attempt_id)
            if entry is None:
                return False
            if entry.get("state") != "cancelled":
                return False
            return self._write_late_outcome_unlocked(
                entry, normalized, latency_ms, failure_category
            )

    def mark_worker_started(self, attempt_id: str) -> bool:
        """Record that the worker thread actually picked up one attempt.

        Called from the worker thread itself BEFORE the blocking socket
        operation begins (kodmial/aa#431). The asyncio-shim cancel path can
        mark the wrapped future cancelled while the thread is genuinely
        still running, so only the thread's own flag distinguishes a clean
        cancellation from an orphan-thread risk.
        """
        with self._lock:
            entry = self._find(attempt_id)
            if entry is None or entry.get("worker_started") is True:
                return False
            entry["worker_started"] = True
            return True

    def claim_worker_start(self, attempt_id: str) -> bool:
        """Atomically claim one attempt for the worker thread.

        Returns ``False`` when another claim already started the thread or
        when a clean cancellation won the race first. In the latter case the
        worker must return without issuing its blocking request, keeping the
        ledger's clean-cancel verdict truthful even if the executor runs the
        queued work item after the wrapped future was cancelled.
        """
        with self._lock:
            entry = self._find(attempt_id)
            if entry is None:
                return True
            if entry.get("worker_started") is True:
                return False
            if entry.get("clean_worker_cancel") is True and entry.get("worker_done") is True:
                return False
            entry["worker_started"] = True
            return True

    def worker_started(self, attempt_id: str) -> bool:
        """Whether the worker thread began executing one attempt."""
        with self._lock:
            entry = self._find(attempt_id)
            return entry is not None and entry.get("worker_started") is True

    def mark_clean_cancel_if_not_started(self, attempt_id: str) -> bool:
        """Atomically record a clean cancellation when no worker started.

        The check and the write happen under the ledger lock, so a worker
        that wins the race first is never misreported as a clean cancel, and
        a clean-cancel winner prevents a later worker claim from issuing the
        request.
        """
        with self._lock:
            entry = self._find(attempt_id)
            if entry is None or entry.get("worker_done") is True:
                return False
            if entry.get("worker_started") is True:
                return False
            entry["worker_done"] = True
            entry["clean_worker_cancel"] = True
            return True

    def mark_worker_done(
        self,
        attempt_id: str,
        *,
        outcome: str = "",
        latency_ms: float | None = None,
        failure_category: str = "",
        clean_cancel: bool = False,
    ) -> bool:
        """Record the worker thread's observable fate for one attempt.

        Called from the worker future's done-callback after either a clean
        cancellation before the thread started (``clean_cancel=True``, no
        orphan thread exists) or a late completion after the awaiting
        coroutine was abandoned. Only such corroboration can prove a
        ``cancelled`` attempt is quiesced. Late-outcome writes are idempotent
        and share one locked field update with the loop-side cancellation
        path, so the first terminal verdict wins consistently.
        """
        with self._lock:
            entry = self._find(attempt_id)
            if entry is None or entry.get("worker_done") is True:
                return False
            entry["worker_done"] = True
            if clean_cancel:
                entry["clean_worker_cancel"] = True
                return True
            outcome_state = _truncate(outcome, _MAX_FIELD_LEN)
            if outcome_state in LATE_OUTCOMES:
                self._write_late_outcome_unlocked(
                    entry, outcome_state, latency_ms, failure_category
                )
            return True

    def entries(self) -> tuple[dict[str, Any], ...]:
        """Return a privacy-safe copy of the attempt recordings."""
        with self._lock:
            return tuple(dict(entry) for entry in self._entries)

    def state_for(self, attempt_id: str) -> str:
        """Return the current state of one attempt (``""`` when unknown)."""
        with self._lock:
            entry = self._find(attempt_id)
            return str(entry.get("state", "")) if entry else ""

    def mark_error_kind(self, attempt_id: str, error_kind: str) -> bool:
        """Attach the bounded classified error kind to one failed attempt."""
        with self._lock:
            entry = self._find(attempt_id)
            if entry is None:
                return False
            entry["error_kind"] = str(error_kind or "")[:64]
            return True

    def _newest_attempt_unlocked(self, causality_id: str | None = None) -> dict[str, Any] | None:
        wanted = _truncate(causality_id, _MAX_CAUSALITY_LEN) if causality_id else ""
        for entry in reversed(self._entries):
            if wanted and str(entry.get("causality_id", "")) != wanted:
                continue
            return dict(entry)
        return None

    def newest_attempt(self, causality_id: str | None = None) -> dict[str, Any] | None:
        """Return the newest attempt record, optionally for one causality id."""
        with self._lock:
            return self._newest_attempt_unlocked(causality_id)

    def newest_terminal_state(self, causality_id: str | None = None) -> str:
        """Return the newest attempt's state (``none`` when nothing recorded).

        ``in-flight`` is a real, observable state here: an attempt whose
        ``attempt-start`` exists without a terminal record is genuinely
        still running inside a worker thread.
        """
        with self._lock:
            entry = self._newest_attempt_unlocked(causality_id)
        if entry is None:
            return "none"
        return str(entry.get("state", IN_FLIGHT_STATE))

    def proven_quiesced(self, causality_id: str | None = None) -> bool:
        """Whether the newest attempt provably cannot leak work anymore.

        ``completed``/``failed``/``aborted`` imply the awaiting coroutine
        observed the outcome on a finished worker. ``cancelled`` proves
        nothing about the HTTP thread until the worker future itself
        terminally completed either cleanly (before the thread started) or
        with a recorded late outcome. ``in-flight`` and unknown attempts
        are never proven quenched.
        """
        with self._lock:
            entry = self._newest_attempt_unlocked(causality_id)
            if entry is None:
                return False
            state = str(entry.get("state", IN_FLIGHT_STATE))
            if state in QUIESCENCE_STATES:
                return True
            if state == "cancelled":
                return entry.get("worker_done") is True
            return False

    def has_unterminated(self, causality_id: str | None = None) -> bool:
        """Whether any attempt (or the newest one) still lacks a terminal state."""
        wanted = _truncate(causality_id, _MAX_CAUSALITY_LEN) if causality_id else ""
        with self._lock:
            for entry in reversed(self._entries):
                if wanted and str(entry.get("causality_id", "")) != wanted:
                    continue
                return str(entry.get("state", IN_FLIGHT_STATE)) == IN_FLIGHT_STATE
            return False

    def in_flight_count(self) -> int:
        """Count attempts that have not reached a terminal state yet."""
        with self._lock:
            return sum(1 for entry in self._entries if entry.get("state") == IN_FLIGHT_STATE)


__all__ = [
    "IN_FLIGHT_STATE",
    "LATE_OUTCOMES",
    "QUIESCENCE_STATES",
    "TERMINAL_ATTEMPT_STATES",
    "AttemptLedger",
]
