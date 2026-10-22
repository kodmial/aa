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

    ``audit`` is the OpenCode client's privacy-safe per-request latency
    audit. The classification is honest about observability: the pinned
    synchronous request API exposes only local completion/failure — not
    server-side streaming — so whenever no request completed the report
    says ``progress-unobservable`` instead of guessing an upstream cause.
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
    completed = [item for item in matched if item.get("success") is True]
    if completed:
        state = "request-completed"
    else:
        in_flight = [item for item in matched if item.get("success") is not False]
        if in_flight:
            # No terminal failure recorded: the request may still be in
            # socket silence. Honest classification keeps progress
            # unobservable instead of guessing a failure cause.
            state = "unconfirmed-in-flight"
        elif all(str(item.get("failure_category", "")) == "cancelled" for item in matched):
            state = "request-cancelled"
        else:
            state = "request-failed"
    last = dict(matched[-1])
    return {
        "request_started": True,
        "request_state": state,
        "http_request_count": len(matched),
        "last_confirmed_request": str(last.get("operation", ""))[:48],
        "last_request_failure_category": str(last.get("failure_category", ""))[:48],
        "last_request_latency_ms": last.get("latency_ms"),
        _OBSERVABILITY: _PROGRESS_UNOBSERVABLE,
    }


def build_timeout_attribution(
    *,
    causality_id: str,
    family: str,
    last_active_stage: str,
    stage_changed_at: float | None,
    started_monotonic: float,
    request_audit: Sequence[object],
    timeout_reason: str,
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
    return attribution


class QuiesceSeamProtocol:
    """Duck-typed Telegram network seam contract for quiescence."""

    quarantined_texts: list[dict[str, Any]]

    def begin_quiescence(self, reason: str) -> None: ...
    def end_quiescence(self) -> None: ...


async def quiesce_and_abort_stalled_turn(
    *,
    app: Any,
    chat_id: int,
    seam: Any,
    grace_s: float,
    reason: str,
    restart: Callable[[], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Stop, cancel and isolate one abandoned scenario turn (kodmial/aa#417).

    Sequence (bounded and deterministic):

    1. begin quiescence: late deliveries from the abandoned turn are
       diverted to the seam quarantine and can never satisfy the next
       scenario's wait;
    2. cancel the per-chat dispatcher worker and drain its queue, awaiting
       shutdown for ``grace_s`` (the OpenCode client issues a best-effort
       ``POST /session/{id}/abort`` on cancellation);
    3. when the turn is non-cooperative (blocking thread survived the
       grace), retire the runtime via ``restart`` so no task in the
       process can deliver stale output afterwards.

    The summary is fixed-vocabulary and privacy-safe. Never raises:
    recovery must not crash the lane itself.
    """
    cleanup: dict[str, Any] = {"reason": str(reason)[:32], "chat_cancelled": False}
    try:
        if seam is not None:
            seam.begin_quiescence(reason)
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
    "build_timeout_attribution",
    "derive_timeout_category",
    "quiesce_and_abort_stalled_turn",
    "request_state_from_audit",
]
