"""Stalled-turn lifecycle repair for the live Gate C campaign (kodmial/aa#417).

The confirmed 2026-10-22 incident (run 39715421348): three serial 30-minute
``no-real-progress`` stalls each left their abandoned turn running while the
scenario loop advanced, so late replies from scenario N contaminated
scenario N+1, deadlines 15-18 were burned on instant budget exhaustion, and
the watchdog timeout stayed ``unknown`` because the child had already
cleared the active stage. This module owns the typed stall recovery:

- privacy-safe attribution: derive the fix-vocabulary timeout category
  from the actually observed last active stage and the OpenCode request
  audit, never from an arbitrary wall-clock cutoff;
- no orphan work: quarantine late deliveries, cancel the abandoned
  per-chat dispatcher turn, await a bounded shutdown, and retire/restart
  the isolated runtime when the turn is non-cooperative. The next
  scenario starts from a quiescent runtime and never shares a stale
  Telegram queue or in-flight graph task;
- bounded recovery: a responsive slow model is never interrupted early;
  recovery only runs after the real no-progress/total-budget trigger.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

logger = logging.getLogger("aa.qualification.live_turn_repair")

STUCK_PROVIDER_STAGES = frozenset(
    {
        "planner",
        "answer",
        "verifier",
        "opencode",
        "provider",
        "whole_turn_judge",
    }
)
STUCK_RETRIEVAL_STAGES = frozenset({"retrieval", "index"})

_TIMEOUT_CATEGORY = "timeout_category"
_LAST_ACTIVE_STAGE = "last_active_stage"
_OBSERVABILITY = "request_progress_observable"
_PROGRESS_UNOBSERVABLE = "progress-unobservable"

# Closed stall-cause vocabulary (kodmial/aa#431): every stalled OpenCode
# operation is attributed to exactly one of these kinds. Heartbeats
# (Telegram typing, SSE events, health probes) are never direct provider
# progress evidence and can therefore never upgrade a classification.
STALL_KIND_HTTP_IN_FLIGHT = "http-in-flight"
STALL_KIND_PROCESS_DEAD = "opencode-process-dead"
STALL_KIND_SERVER_ALIVE_SESSION_UNOBSERVED = "opencode-server-alive-but-session-unobserved"
STALL_KIND_PROVIDER_ERROR_CONFIRMED = "provider-error-confirmed"
STALL_KIND_GRAPH_STALL = "graph-stall"
STALL_KIND_DELIVERY_STALL = "delivery-stall"
STALL_KIND_UNKNOWN = "unknown"
STALL_KINDS: tuple[str, ...] = (
    STALL_KIND_HTTP_IN_FLIGHT,
    STALL_KIND_PROCESS_DEAD,
    STALL_KIND_SERVER_ALIVE_SESSION_UNOBSERVED,
    STALL_KIND_PROVIDER_ERROR_CONFIRMED,
    STALL_KIND_GRAPH_STALL,
    STALL_KIND_DELIVERY_STALL,
    STALL_KIND_UNKNOWN,
)

_TERMINAL_REQUEST_STATES = ("completed", "failed", "cancelled", "aborted")
_ATTEMPT_STATES = ("in-flight", "completed", "failed", "cancelled", "aborted")
# Classified error kinds that prove an actual provider-side error, as
# opposed to a local transport/socket result.
_PROVIDER_ERROR_KINDS = frozenset(
    {"rate-limit", "deterministic", "provider-access", "transient", "session-not-found"}
)
_DELIVERY_STAGES = frozenset({"delivery", "telegram", "seam", "post", "transport"})
_UNKNOWN_STATE = "none"


def derive_timeout_category(stage: str) -> str:
    """Map one observed active stage to the closed-vocabulary category.

    Same vocabulary as the outer watchdog: the stage evidence, not the
    timeout clock, decides provider vs retrieval. An honestly unknown or
    absent stage stays ``unknown`` (kodmial/aa#411 semantics preserved).
    """
    normalized = str(stage or "").strip().lower()
    if not normalized:
        return "unknown"
    if normalized in STUCK_PROVIDER_STAGES:
        return "stuck-provider"
    if normalized in STUCK_RETRIEVAL_STAGES:
        return "stuck-retrieval"
    return "unknown"


def request_state_from_audit(audit: Sequence[object], causality_id: str) -> dict[str, Any]:
    """Classify OpenCode request evidence for one causality id.

    ``audit`` is the OpenCode client's privacy-safe per-request attempt
    audit. The NEWEST attempt always wins: an earlier completed request
    never stands in as proof that the newest (still running) request is
    finished (kodmial/aa#431). Rows recorded by the current client carry
    ``attempt_id`` plus an explicit ``state``; a row that touched the
    ``attempt-start`` boundary with no terminal state is honestly reported
    as ``http-in-flight``. Legacy rows without attempt ids keep the old
    conservative classification. The classification is honest about
    observability: the pinned synchronous request API exposes only local
    completion/failure — not server-side streaming — so when nothing
    directly proves a request state the report says
    ``progress-unobservable`` instead of guessing an upstream cause.
    """
    entries = [item for item in audit if isinstance(item, Mapping)]
    matched = [
        item for item in entries if str(item.get("causality_id", "")) == str(causality_id).strip()
    ]
    if not matched:
        return {
            "request_started": False,
            "request_state": "none",
            "http_request_count": 0,
            "last_confirmed_request": "",
            _OBSERVABILITY: _PROGRESS_UNOBSERVABLE,
        }
    attempt_rows = [(index, item) for index, item in enumerate(matched) if item.get("attempt_id")]
    if attempt_rows:
        ordered = sorted(
            attempt_rows,
            key=lambda pair: (pair[1].get("attempt_started_monotonic", pair[0]), pair[0]),
        )
        _index, newest = ordered[-1]
        state = str(newest.get("state", "in-flight"))
        if state == "in-flight":
            state_name = "http-in-flight"
        elif state == "completed":
            state_name = "request-completed"
        elif state == "failed":
            state_name = "request-failed"
        elif state == "cancelled":
            state_name = "request-cancelled"
        elif state == "aborted":
            state_name = "request-aborted"
        else:
            state_name = _legacy_request_state(matched)
        last = dict(newest)
    else:
        state_name = _legacy_request_state(matched)
        last = dict(matched[-1])
    return {
        "request_started": True,
        "request_state": state_name,
        "http_request_count": len(matched),
        "last_confirmed_request": str(last.get("operation", ""))[:48],
        "last_request_state": str(last.get("state", ""))[:48],
        "last_attempt_id": str(last.get("attempt_id", ""))[:64],
        "late_outcome": str(last.get("late_outcome", ""))[:32],
        "last_request_failure_category": str(last.get("failure_category", ""))[:48],
        "last_request_error_kind": str(last.get("error_kind", ""))[:48],
        "last_request_latency_ms": last.get("latency_ms"),
        _OBSERVABILITY: _PROGRESS_UNOBSERVABLE,
    }


def _legacy_request_state(matched: Sequence[Mapping[str, Any]]) -> str:
    """Pre-#431 classification for audit rows without attempt ids."""
    completed = [item for item in matched if item.get("success") is True]
    if completed:
        return "request-completed"
    in_flight = [item for item in matched if item.get("success") is not False]
    if in_flight:
        # No terminal failure recorded: the request may still be in
        # socket silence. Honest classification keeps progress
        # unobservable instead of guessing a failure cause.
        return "unconfirmed-in-flight"
    if all(str(item.get("failure_category", "")) == "cancelled" for item in matched):
        return "request-cancelled"
    return "request-failed"


def classify_stall_kind(
    *,
    causality_id: str,
    request_audit: Sequence[object],
    last_active_stage: str = "",
    process_alive: bool | None = None,
    session_progress_observed: bool | None = None,
) -> str:
    """Attribute one stalled turn to exactly one closed stall kind.

    The decision uses DIRECT evidence only:

    - a confirmed provider-side classified error overrides everything;
    - if the OpenCode process is dead, the stall is
      ``opencode-process-dead`` (a terminated worker cannot have a genuine
      HTTP request in flight);
    - a live ``attempt-start`` row without a terminal state is
      ``http-in-flight`` (the HTTP request is provably still running);
    - a request that locally finished without any session progress
      (server alive) is ``opencode-server-alive-but-session-unobserved``;
    - a completed request whose graph never advanced is ``graph-stall``
      or, at the last delivery stage, ``delivery-stall``.

    Heartbeats (typing, SSE, health probes) are never passed in as
    ``session_progress_observed``: they are not provider token progress.
    An unset ``session_progress_observed`` (``None``) is treated exactly
    like ``False`` on purpose: progress stays unproven unless positively
    observed. An honestly unknown combination stays ``unknown``.
    """
    report = request_state_from_audit(request_audit, causality_id)
    request_state = str(report.get("request_state", ""))
    newest_state = str(report.get("last_request_state", ""))
    provider_error = (
        newest_state == "failed"
        and str(report.get("last_request_error_kind", "")) in _PROVIDER_ERROR_KINDS
    )
    if provider_error:
        return STALL_KIND_PROVIDER_ERROR_CONFIRMED
    if process_alive is False:
        return STALL_KIND_PROCESS_DEAD
    if request_state == "http-in-flight":
        return STALL_KIND_HTTP_IN_FLIGHT
    if request_state == "none" or request_state == "unconfirmed-in-flight":
        return STALL_KIND_UNKNOWN
    if session_progress_observed is not True:
        return STALL_KIND_SERVER_ALIVE_SESSION_UNOBSERVED
    if request_state == "request-completed":
        normalized = str(last_active_stage or "").strip().lower()
        if normalized in _DELIVERY_STAGES:
            return STALL_KIND_DELIVERY_STALL
        return STALL_KIND_GRAPH_STALL
    return STALL_KIND_UNKNOWN


def build_timeout_attribution(
    *,
    causality_id: str,
    family: str,
    last_active_stage: str,
    stage_changed_at: float | None,
    started_monotonic: float,
    request_audit: Sequence[object],
    timeout_reason: str,
    process_alive: bool | None = None,
    session_progress_observed: bool | None = None,
) -> dict[str, Any]:
    """Build the privacy-safe stalled-turn attribution block."""
    elapsed_ms = max(0.0, time.monotonic() - started_monotonic) * 1000.0
    attribution: dict[str, Any] = {
        "causality_id": str(causality_id)[:64],
        "family": str(family)[:64],
        _TIMEOUT_CATEGORY: derive_timeout_category(last_active_stage),
        "timeout_reason": str(timeout_reason or "unknown")[:32],
        _LAST_ACTIVE_STAGE: str(last_active_stage)[:64],
        "stage_elapsed_ms": (
            round(max(0.0, time.monotonic() - stage_changed_at) * 1000.0, 1)
            if stage_changed_at is not None
            else None
        ),
        "delivery_latency_ms": round(elapsed_ms, 1),
        _OBSERVABILITY: _PROGRESS_UNOBSERVABLE,
    }
    attribution.update(request_state_from_audit(request_audit, causality_id))
    attribution[_OBSERVABILITY] = _PROGRESS_UNOBSERVABLE
    attribution["stall_kind"] = classify_stall_kind(
        causality_id=causality_id,
        request_audit=request_audit,
        last_active_stage=last_active_stage,
        process_alive=process_alive,
        session_progress_observed=session_progress_observed,
    )
    if attribution.get("stall_kind") == STALL_KIND_UNKNOWN:
        attribution["stall_kind_uncertain"] = True
    return attribution


class QuiesceSeamProtocol:
    """Duck-typed Telegram network seam contract for quiescence."""

    quarantined_texts: list[dict[str, Any]]

    def begin_quiescence(self, reason: str) -> None: ...
    def end_quiescence(self) -> None: ...


async def _evaluate_quiescence_probe(
    probe: Callable[[], Any] | None,
) -> bool:
    """Evaluate the quiescence probe lazily, after cancellation has run.

    Supported probe result forms: falsy (not proven), truthy, or an
    awaitable resolving to one of those. Any probe failure degrades to
    ``False`` (not proven) so the fail-closed isolation path wins.
    """
    if probe is None:
        return False
    try:
        result = probe()
    except Exception:
        return False
    if inspect.isawaitable(result):
        try:
            result = await result
        except Exception:
            return False
    return bool(result)


async def quiesce_and_abort_stalled_turn(
    *,
    app: Any,
    chat_id: int,
    seam: Any,
    grace_s: float,
    reason: str,
    restart: Callable[[], Awaitable[None]] | None = None,
    quiescence_probe: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Stop, cancel and isolate one abandoned scenario turn (kodmial/aa#417).

    Sequence (bounded and deterministic):

    1. begin quiescence: late deliveries from the abandoned turn are
       diverted to the seam quarantine and can never satisfy the next
       scenario's wait;
    2. cancel the per-chat dispatcher worker and drain its queue, awaiting
       shutdown for ``grace_s`` (the OpenCode client issues a best-effort
       ``POST /session/{id}/abort`` on cancellation);
    3. when the turn is non-cooperative, OR the abandoned turn's HTTP
       attempt is not PROVEN quiesced, retire the runtime via ``restart``
       so no task or worker thread in the process can deliver stale
       output afterwards (kodmial/aa#431): ``asyncio.Task.done()`` proves
       nothing about the underlying HTTP thread; the attempt ledger's
       proven-quiescence verdict (or the absence of any in-flight
       attempt) is the only acceptable proof. When the caller cannot
       supply proof, the runtime is isolated by default — a late response
       can never satisfy a different scenario.

    The summary is fixed-vocabulary and privacy-safe. Never raises:
    recovery must not crash the lane itself.
    """
    cleanup: dict[str, Any] = {"reason": str(reason)[:32], "chat_cancelled": False}
    quiesced = False
    try:
        if seam is not None:
            seam.begin_quiescence(reason)
            quiesced = True
            cleanup["quiesced"] = True
        try:
            result = await app.cancel_pending_turns(int(chat_id), grace_s=grace_s)
        except Exception:
            result = {"had_worker": False, "stopped_within_grace": True}
        if isinstance(result, Mapping):
            cleanup.update({key: value for key, value in result.items()})
            cleanup["chat_cancelled"] = bool(result.get("had_worker"))
        # Bounded settle for any delivery already inside the seam when the
        # cancellation landed; everything is quarantined above.
        await _settle(grace_s)
        in_flight = False
        try:
            in_flight = bool(app.turn_in_flight(int(chat_id)))
        except Exception:
            in_flight = False
        if not cleanup.get("non_cooperative") and in_flight:
            cleanup["non_cooperative"] = True
        http_quiescence_proven = await _evaluate_quiescence_probe(quiescence_probe)
        cleanup["http_quiescence_proven"] = http_quiescence_proven
        if http_quiescence_proven is not True:
            # Task.done() never proves the HTTP thread stopped (kodmial/aa#431).
            # An absent or failed probe counts as NOT proven: the runtime is
            # isolated by default so no orphan worker can leak stale output.
            cleanup["orphan_http_attempt_isolated"] = True
            cleanup["non_cooperative"] = True
        if cleanup.get("non_cooperative") and restart is not None:
            try:
                cleanup["runtime_retired"] = True
                await restart()
                try:
                    cleanup["runtime_restarted"] = bool(app.running)
                except Exception:
                    cleanup["runtime_restarted"] = True
            except Exception:
                cleanup["runtime_restarted"] = False
        if seam is not None:
            try:
                cleanup["quarantined_delivery_count"] = len(seam.quarantined_texts)
            except Exception:
                pass
    except Exception as exc:
        logger.warning(
            "stall cleanup failed",
            extra={
                "chat_id": chat_id,
                "reason": str(reason)[:32],
                "error": type(exc).__name__,
            },
        )
        cleanup["cleanup_failed"] = True
        cleanup["cleanup_error"] = type(exc).__name__
    finally:
        # Quarantine is scoped to this cleanup pass. Always release it,
        # including on failure and when the stalled scenario is the last
        # one, so the lane is never left fenced indefinitely.
        if quiesced:
            try:
                seam.end_quiescence()
                cleanup["quiescence_released"] = True
            except Exception:
                cleanup["quiescence_released"] = False
    return cleanup


_SETTLE_CAP_S = 1.0


async def _settle(grace_s: float) -> None:
    """Bounded yield so queued seam deliveries observably reconcile.

    The cap is deliberately larger than a single event-loop tick: late
    deliveries already inside the Telegram seam need time to reach the
    quarantine before the next scenario starts. It stays well below the
    scenario grace so it can never meaningfully extend a stall.
    """
    import asyncio

    if grace_s <= 0:
        await asyncio.sleep(0)
        return
    await asyncio.sleep(min(grace_s, _SETTLE_CAP_S))


__all__ = [
    "STALL_KINDS",
    "STALL_KIND_DELIVERY_STALL",
    "STALL_KIND_GRAPH_STALL",
    "STALL_KIND_HTTP_IN_FLIGHT",
    "STALL_KIND_PROCESS_DEAD",
    "STALL_KIND_PROVIDER_ERROR_CONFIRMED",
    "STALL_KIND_SERVER_ALIVE_SESSION_UNOBSERVED",
    "STALL_KIND_UNKNOWN",
    "build_timeout_attribution",
    "classify_stall_kind",
    "derive_timeout_category",
    "quiesce_and_abort_stalled_turn",
    "request_state_from_audit",
]
