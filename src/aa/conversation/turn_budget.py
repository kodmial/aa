"""Typed turn-level deadline and stage budget contract (kodmial/aa#335).

The 5 second ``INTERACTIVE_LATENCY_BUDGET_MS`` in
:mod:`aa.retrieval.evidence` is a diagnostic target for the warm local
RRF-only RAM retrieval stage (BM25 + E5/FAISS + RRF fusion over the
in-memory index). It is CPU work with no network dependency and must
never serve as the total sequential deadline for LLM/provider semantic
selection or full-book coverage reads.

This module owns the real turn deadline instead:

- :class:`TurnBudget` is created once per turn (by the compiled graph,
  effectively at retrieval entry with upstream planner cost folded in)
  and shared by discovery, model selection, exact canonical reads,
  coverage assessment, answer generation, verification and transport.
- Model (provider) awaits receive the *remaining end-to-end* budget
  (authoritative Gate E policies: hard max 120s, p95 target 60s, graph
  turn budget 105s with delivery margin), minus a small downstream
  reserve for answer/verifier minimal slices. No fixed 5s/8s cutoff is
  silently applied to genuinely successful model coverage.
- The local RRF stage is measured against its 5s diagnostic target and
  reports a ``local_retrieval_slow`` stage flag; slowness there never
  misclassifies adequate LLM evidence as exhausted.
- ``asyncio.to_thread`` hybrid search is noncancelable: ``wait_for``
  cancellation only cancels the waiter, never the synchronous thread
  work. Local retrieval therefore runs on one bounded single-worker
  executor with explicit worker-in-flight accounting; a waiter timeout
  reports ``worker_in_flight`` truthfully without claiming cancellation
  occurred, and no unbounded queued orphan work accumulates.
- Typed stages (``local_retrieval_slow``, ``provider_semantic_timeout``,
  ``provider_429``, ``coverage_insufficient``, ``verifier_timeout``,
  ``transport_unconfirmed``) carry anonymized per-stage durations plus
  source/need hashes and remaining budget. No book/user text is stored.

An exhausted semantic model call can never pass as sufficient book
evidence: timeouts raise typed errors that the retrieval loop converts
to ``coverage_status=exhausted`` with the typed stage preserved.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# Diagnostic target for the warm local RRF-only RAM stage. Must stay
# equal to ``aa.retrieval.evidence.INTERACTIVE_LATENCY_BUDGET_MS``;
# enforced by tests. Diagnostic only: never a model-call deadline.
LOCAL_RRF_DIAGNOSTIC_BUDGET_MS = 5000.0

# Real turn deadline owned by the graph (seconds). Must stay equal to
# ``aa.conversation.turn_pipeline.TURN_END_TO_END_BUDGET_S`` (105s <
# 120s hard SLO, delivery margin kept); enforced by tests.
TURN_END_TO_END_BUDGET_S = 105.0

# Authoritative unchanged Gate E policies (issue #146): hard max 120s,
# p95 target 60s. Duplicated here so this leaf module never imports the
# qualification DAG; equality is enforced by tests.
TURN_HARD_SLO_MS = 120_000.0
TURN_P95_TARGET_MS = 60_000.0

# Provider transport ceiling (model_adapter request_timeout). A model
# await never exceeds this even when the turn has more remaining.
PROVIDER_REQUEST_TIMEOUT_S = 120.0

# Minimum useful slices kept for downstream answer/verifier stages when
# allocating a semantic timeout from the remaining turn budget. Below
# the answer slice no answer round starts; below the verifier slice no
# verifier round starts (turn_pipeline minimal slices: 1s + 3s + margin).
DOWNSTREAM_MIN_RESERVE_S = 5.0

# Floor for one allocated model await so a nearly-spent budget still
# fails fast instead of launching a zero-time round.
MIN_MODEL_SLICE_S = 0.05

# Typed stage names for per-stage outcomes (privacy-safe, no text).
STAGE_LOCAL_RETRIEVAL_SLOW = "local_retrieval_slow"
STAGE_PROVIDER_SEMANTIC_TIMEOUT = "provider_semantic_timeout"
STAGE_PROVIDER_429 = "provider_429"
STAGE_COVERAGE_INSUFFICIENT = "coverage_insufficient"
STAGE_VERIFIER_TIMEOUT = "verifier_timeout"
STAGE_TRANSPORT_UNCONFIRMED = "transport_unconfirmed"

TYPED_STAGES: tuple[str, ...] = (
    STAGE_LOCAL_RETRIEVAL_SLOW,
    STAGE_PROVIDER_SEMANTIC_TIMEOUT,
    STAGE_PROVIDER_429,
    STAGE_COVERAGE_INSUFFICIENT,
    STAGE_VERIFIER_TIMEOUT,
    STAGE_TRANSPORT_UNCONFIRMED,
)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def hash_ids(ids: list[str]) -> str:
    """Return an anonymized digest for id lists (never the ids in full)."""
    cleaned = sorted(str(item).strip() for item in (ids or []) if str(item).strip())
    return _short_hash("|".join(cleaned))


class TurnBudgetExpired(TimeoutError):
    """Remaining end-to-end turn budget is spent (typed, fail-closed)."""

    def __init__(
        self,
        message: str = "turn deadline exceeded",
        *,
        stage: str = STAGE_PROVIDER_SEMANTIC_TIMEOUT,
        elapsed_ms: float = 0.0,
        remaining_ms: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.elapsed_ms = float(elapsed_ms)
        self.remaining_ms = float(remaining_ms)


class ProviderSemanticTimeout(TimeoutError):
    """One semantic model await exceeded the remaining turn budget."""

    def __init__(
        self,
        message: str = "semantic provider call exceeded turn deadline",
        *,
        stage: str = STAGE_PROVIDER_SEMANTIC_TIMEOUT,
        elapsed_ms: float = 0.0,
        allocated_s: float = 0.0,
        remaining_ms: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.elapsed_ms = float(elapsed_ms)
        self.allocated_s = float(allocated_s)
        self.remaining_ms = float(remaining_ms)


class LocalRetrievalTimeout(TimeoutError):
    """Local RRF worker waiter expired; thread work may still be in flight."""

    def __init__(
        self,
        message: str = "local retrieval worker deadline exceeded",
        *,
        elapsed_ms: float = 0.0,
        worker_in_flight: bool = False,
    ) -> None:
        super().__init__(message)
        self.elapsed_ms = float(elapsed_ms)
        self.worker_in_flight = bool(worker_in_flight)


@dataclass
class StageRecord:
    """One anonymized per-stage duration (no book/user text)."""

    stage: str
    ok: bool
    latency_ms: float
    category: str = ""
    id_digest: str = ""
    remaining_ms: float = 0.0


@dataclass
class TurnBudget:
    """Typed per-turn deadline shared across all graph stages.

    ``started_monotonic`` uses ``time.perf_counter`` so fake-clock tests
    can drive it deterministically. ``upstream_spent_s`` folds in graph
    cost already paid before this budget was created (planner wall
    clock); direct retrieval calls pass ``0.0``.
    """

    started_monotonic: float = field(default_factory=time.perf_counter)
    deadline_s: float = TURN_END_TO_END_BUDGET_S
    upstream_spent_s: float = 0.0
    stages: list[StageRecord] = field(default_factory=list)

    def elapsed_s(self) -> float:
        try:
            return max(0.0, float(self.upstream_spent_s)) + max(
                0.0, time.perf_counter() - float(self.started_monotonic)
            )
        except Exception:
            return max(0.0, float(self.upstream_spent_s or 0.0))

    def elapsed_ms(self) -> float:
        return self.elapsed_s() * 1000.0

    def remaining_s(self) -> float:
        try:
            return float(self.deadline_s) - self.elapsed_s()
        except Exception:
            return 0.0

    def remaining_ms(self) -> float:
        return self.remaining_s() * 1000.0

    def is_expired(self) -> bool:
        return self.remaining_s() <= 0.0

    def allocate_semantic_timeout_s(self, *, reserve_s: float = DOWNSTREAM_MIN_RESERVE_S) -> float:
        """Timeout for one semantic model await from remaining turn budget.

        The allocation is the remaining end-to-end budget minus the small
        downstream reserve for answer/verifier minimal slices, clamped to
        the provider transport ceiling. There is deliberately no fixed
        5s/8s cutoff: a genuinely successful 4s selector plus 4s coverage
        pair (total far below the turn bound) is never forcibly
        exhausted, while a call past the real remaining deadline still
        fails fast to a typed timeout.
        """
        try:
            reserve = max(0.0, float(reserve_s))
        except (TypeError, ValueError):
            reserve = DOWNSTREAM_MIN_RESERVE_S
        remaining = self.remaining_s() - reserve
        if remaining <= 0:
            raise TurnBudgetExpired(
                "turn deadline already spent",
                stage=STAGE_PROVIDER_SEMANTIC_TIMEOUT,
                elapsed_ms=self.elapsed_ms(),
                remaining_ms=self.remaining_ms(),
            )
        return max(MIN_MODEL_SLICE_S, min(float(remaining), PROVIDER_REQUEST_TIMEOUT_S))

    def record_stage(
        self,
        *,
        stage: str,
        ok: bool,
        latency_ms: float,
        category: str = "",
        id_digest: str = "",
    ) -> StageRecord:
        record = StageRecord(
            stage=str(stage),
            ok=bool(ok),
            latency_ms=max(0.0, float(latency_ms or 0.0)),
            category=str(category or "")[:64],
            id_digest=str(id_digest or "")[:32],
            remaining_ms=self.remaining_ms(),
        )
        self.stages.append(record)
        return record

    def snapshot(self) -> dict[str, Any]:
        """Privacy-safe snapshot: durations, digests, remaining (no text)."""
        return {
            "deadline_s": float(self.deadline_s),
            "elapsed_ms": round(self.elapsed_ms(), 3),
            "remaining_ms": round(self.remaining_ms(), 3),
            "upstream_spent_s": round(float(self.upstream_spent_s or 0.0), 3),
            "stages": [
                {
                    "stage": item.stage,
                    "ok": bool(item.ok),
                    "latency_ms": round(float(item.latency_ms), 3),
                    "category": item.category,
                    "id_digest": item.id_digest,
                    "remaining_ms": round(float(item.remaining_ms), 3),
                }
                for item in self.stages[-32:]
            ],
        }


def new_turn_budget(
    *, upstream_latency_ms: float = 0.0, deadline_s: float | None = None
) -> TurnBudget:
    """Create one turn budget folding in already-spent upstream cost."""
    try:
        upstream_s = max(0.0, float(upstream_latency_ms or 0.0)) / 1000.0
    except (TypeError, ValueError):
        upstream_s = 0.0
    try:
        deadline = float(deadline_s) if deadline_s is not None else TURN_END_TO_END_BUDGET_S
    except (TypeError, ValueError):
        deadline = TURN_END_TO_END_BUDGET_S
    if not deadline > 0:
        deadline = TURN_END_TO_END_BUDGET_S
    return TurnBudget(
        started_monotonic=time.perf_counter(),
        deadline_s=deadline,
        upstream_spent_s=upstream_s,
    )


def turn_budget_from_state(state: Any) -> TurnBudget:
    """Build the graph-owned budget from orchestration state (no text).

    Folds the planner wall clock (``retry_state.planner_latency_ms``)
    into upstream cost so retrieval model awaits receive the realistic
    remaining end-to-end budget instead of a fixed local cutoff.
    """
    upstream_ms = 0.0
    try:
        retry = (state.get("retry_state", {}) or {}) if isinstance(state, dict) else {}
        retry_d = dict(retry) if isinstance(retry, dict) else {}
        upstream_ms = max(0.0, float(retry_d.get("planner_latency_ms", 0.0) or 0.0))
    except (TypeError, ValueError, AttributeError):
        upstream_ms = 0.0
    return new_turn_budget(upstream_latency_ms=upstream_ms)


async def await_model_under_turn_budget(
    coro_factory: Any,
    budget: TurnBudget,
    *,
    stage: str = STAGE_PROVIDER_SEMANTIC_TIMEOUT,
    id_digest: str = "",
    reserve_s: float = DOWNSTREAM_MIN_RESERVE_S,
) -> Any:
    """Await one provider/model coroutine under the turn deadline.

    ``coro_factory`` is a zero-argument callable producing the awaitable,
    invoked only after the budget check so an already-spent budget never
    creates an un-awaited coroutine. Timeouts raise
    :class:`ProviderSemanticTimeout` (typed ``provider_semantic_timeout``
    with anonymized durations); ``asyncio.CancelledError`` and provider
    429 (``OpenCodeRateLimitError``) always propagate and are recorded as
    ``provider_429``, never converted to exhaustion. Any other provider
    failure propagates to the caller (coverage maps it to conservative
    uncovered, never fake sufficiency).
    """
    timeout_s = budget.allocate_semantic_timeout_s(reserve_s=reserve_s)
    started = time.perf_counter()
    coro = coro_factory() if callable(coro_factory) else coro_factory
    try:
        from aa.opencode.errors import OpenCodeRateLimitError as _RateLimitCls

        _rate_limit_types: Any = (_RateLimitCls,)
    except ImportError:
        _rate_limit_types = ()
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_s)
    except _rate_limit_types:
        # Dedicated 429 path: OpenCodeRateLimitError is a plain Exception
        # (never a TimeoutError), so it must be caught before the timeout
        # handlers below. Recorded as provider_429 and never converted to
        # exhaustion; always propagates for runner lifecycle recovery.
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        budget.record_stage(
            stage=STAGE_PROVIDER_429,
            ok=False,
            latency_ms=elapsed_ms,
            category="rate-limit",
            id_digest=id_digest,
        )
        raise
    except asyncio.CancelledError as exc:
        if not isinstance(exc, TimeoutError):
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        budget.record_stage(
            stage=stage,
            ok=False,
            latency_ms=elapsed_ms,
            category="deadline-exceeded",
            id_digest=id_digest,
        )
        raise ProviderSemanticTimeout(
            "semantic provider call exceeded turn deadline",
            stage=stage,
            elapsed_ms=elapsed_ms,
            allocated_s=timeout_s,
            remaining_ms=budget.remaining_ms(),
        ) from exc
    except TimeoutError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        budget.record_stage(
            stage=stage,
            ok=False,
            latency_ms=elapsed_ms,
            category="deadline-exceeded",
            id_digest=id_digest,
        )
        raise ProviderSemanticTimeout(
            "semantic provider call exceeded turn deadline",
            stage=stage,
            elapsed_ms=elapsed_ms,
            allocated_s=timeout_s,
            remaining_ms=budget.remaining_ms(),
        ) from exc
    except Exception:
        # Any other provider failure propagates untouched (the dedicated
        # 429 branch above already handled rate limits); coverage maps it
        # to conservative uncovered, never fake sufficiency.
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    budget.record_stage(
        stage=stage,
        ok=True,
        latency_ms=elapsed_ms,
        category="served",
        id_digest=id_digest,
    )
    return result


# ---- Bounded local RRF worker -------------------------------------------
# ``asyncio.to_thread`` (and ``run_in_executor``) cannot cancel
# synchronous thread work: ``wait_for`` expiry only cancels the waiter.
# Local hybrid search therefore runs on one dedicated single-worker
# executor with explicit worker accounting. A waiter timeout reports
# ``worker_in_flight=True`` truthfully (the thread keeps running to
# completion and is reaped via its done-callback) instead of claiming
# cancellation occurred. Admission is bounded: at most one worker plus
# one queued waiter, so a slow thread can never accumulate orphan queue
# depth across loop iterations.

_LOCAL_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aa-rrf-local")
_LOCAL_LOCK = threading.Lock()
_LOCAL_WORKER_ACTIVE = 0
_LOCAL_WAITERS = 0
_LOCAL_COMPLETED = 0
_LOCAL_WAITER_TIMEOUTS = 0


def local_worker_info() -> dict[str, int]:
    """Current bounded-worker accounting (counts only, no text)."""
    with _LOCAL_LOCK:
        return {
            "worker_active": int(_LOCAL_WORKER_ACTIVE),
            "waiters": int(_LOCAL_WAITERS),
            "completed": int(_LOCAL_COMPLETED),
            "waiter_timeouts": int(_LOCAL_WAITER_TIMEOUTS),
            "max_workers": 1,
            "max_queue": 1,
        }


def _worker_done_callback(_future: Any) -> None:
    global _LOCAL_WORKER_ACTIVE, _LOCAL_COMPLETED
    with _LOCAL_LOCK:
        _LOCAL_WORKER_ACTIVE = max(0, _LOCAL_WORKER_ACTIVE - 1)
        _LOCAL_COMPLETED += 1


@dataclass(frozen=True)
class LocalRetrievalReport:
    """Outcome accounting for one bounded local RRF run (no text)."""

    elapsed_ms: float
    slow: bool
    timed_out: bool
    worker_in_flight: bool


async def run_local_retrieval_bounded(
    func: Any,
    *args: Any,
    timeout_s: float | None = None,
    diagnostic_ms: float = LOCAL_RRF_DIAGNOSTIC_BUDGET_MS,
    **kwargs: Any,
) -> tuple[Any, LocalRetrievalReport]:
    """Run sync ``func`` on the bounded local worker with waiter timeout.

    The diagnostic threshold only sets the ``slow`` flag
    (``local_retrieval_slow``); it never converts adequate evidence into
    exhaustion. ``timeout_s=None`` waits for the worker (bounded queue
    still applies); a numeric timeout raises
    :class:`LocalRetrievalTimeout` on waiter expiry while the worker
    keeps running to completion in the background (reported via
    ``worker_in_flight``, reaped by its done-callback, never leaked
    unboundedly).
    """
    global _LOCAL_WORKER_ACTIVE, _LOCAL_WAITERS, _LOCAL_WAITER_TIMEOUTS
    loop = asyncio.get_running_loop()
    with _LOCAL_LOCK:
        # _LOCAL_WORKER_ACTIVE counts submitted-but-not-reaped executor
        # items (running + executor-queued); _LOCAL_WAITERS counts
        # asyncio waiters still attached. Each admitted call increments
        # both, so summing them double-counts one call as two and would
        # reject any concurrent second call. Admit on the worker count
        # alone: at most one running worker plus one queued item.
        queued = int(_LOCAL_WORKER_ACTIVE)
        if queued >= 2:
            raise LocalRetrievalTimeout(
                "local retrieval admission bound exceeded",
                elapsed_ms=0.0,
                worker_in_flight=True,
            )
        _LOCAL_WORKER_ACTIVE += 1
        _LOCAL_WAITERS += 1
    started = time.perf_counter()
    try:
        future = loop.run_in_executor(_LOCAL_EXECUTOR, functools.partial(func, *args, **kwargs))
    except Exception:
        with _LOCAL_LOCK:
            _LOCAL_WORKER_ACTIVE = max(0, _LOCAL_WORKER_ACTIVE - 1)
            _LOCAL_WAITERS = max(0, _LOCAL_WAITERS - 1)
        raise
    try:
        future.add_done_callback(_worker_done_callback)
    except Exception:
        pass
    timed_out = False
    try:
        if timeout_s is None:
            result = await future
        else:
            waiter_timeout = max(MIN_MODEL_SLICE_S, float(timeout_s))
            result = await asyncio.wait_for(future, timeout=waiter_timeout)
    except TimeoutError as exc:
        timed_out = True
        with _LOCAL_LOCK:
            _LOCAL_WAITER_TIMEOUTS += 1
        # Waiter expiry never cancels sync thread work: the worker stays
        # in flight until its done-callback reaps it, even if the
        # asyncio wrapper future reports done/cancelled.
        worker_in_flight = True
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # Do not claim cancellation: the sync thread keeps running; its
        # done-callback reaps the worker slot. Detach the waiter only.
        raise LocalRetrievalTimeout(
            "local retrieval waiter deadline exceeded",
            elapsed_ms=elapsed_ms,
            worker_in_flight=worker_in_flight,
        ) from exc
    finally:
        with _LOCAL_LOCK:
            _LOCAL_WAITERS = max(0, _LOCAL_WAITERS - 1)
            if timed_out:
                # Waiter expired: the worker slot stays occupied until the
                # thread finishes and its done-callback decrements it.
                # Compensate the optimistic increment above would
                # double-count; the callback owns the decrement now.
                pass
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    # Success holds the worker result, so no thread work remains in
    # flight. Report False explicitly: deriving this from future.done()
    # is unsound in general (a cancelled waiter wrapper can read done
    # while the sync thread still runs), and the timeout path above
    # already reports True unconditionally for that reason.
    worker_in_flight = False
    try:
        slow = float(elapsed_ms) > float(diagnostic_ms)
    except (TypeError, ValueError):
        slow = False
    return result, LocalRetrievalReport(
        elapsed_ms=float(elapsed_ms),
        slow=bool(slow),
        timed_out=False,
        worker_in_flight=bool(worker_in_flight),
    )


__all__ = [
    "DOWNSTREAM_MIN_RESERVE_S",
    "LOCAL_RRF_DIAGNOSTIC_BUDGET_MS",
    "MIN_MODEL_SLICE_S",
    "PROVIDER_REQUEST_TIMEOUT_S",
    "STAGE_COVERAGE_INSUFFICIENT",
    "STAGE_LOCAL_RETRIEVAL_SLOW",
    "STAGE_PROVIDER_429",
    "STAGE_PROVIDER_SEMANTIC_TIMEOUT",
    "STAGE_TRANSPORT_UNCONFIRMED",
    "STAGE_VERIFIER_TIMEOUT",
    "TURN_END_TO_END_BUDGET_S",
    "TURN_HARD_SLO_MS",
    "TURN_P95_TARGET_MS",
    "TYPED_STAGES",
    "LocalRetrievalReport",
    "LocalRetrievalTimeout",
    "ProviderSemanticTimeout",
    "StageRecord",
    "TurnBudget",
    "TurnBudgetExpired",
    "await_model_under_turn_budget",
    "hash_ids",
    "local_worker_info",
    "new_turn_budget",
    "run_local_retrieval_bounded",
    "turn_budget_from_state",
]
