"""Independent outer watchdog for the live Gate C qualification lane.

The in-process ``asyncio`` deadline in
``product_contract_live.run_live_telegram_evidence_lane`` cannot preempt a
blocking synchronous call inside the event loop (kodmial/aa#411). This module
runs the live lane in a dedicated child process and supervises it from the
parent using wall-clock progress markers, so a hung provider, retrieval call,
or local OpenCode descendant is terminated before the GitHub Actions job
deadline.

Recovery contract:

- per-scenario deadline: the child publishes a deadline when a scenario
  starts; the parent escalates only after that deadline and a small grace;
- total deadline: derived from the Actions job start/timeout when available,
  leaving an explicit finalisation reserve for Gate E/F and artifact upload;
- kill escalation: SIGTERM to the child's process group, then SIGKILL, so
  descendants started by the live runtime are reaped;
- typed negative evidence: completed per-scenario diagnostics are persisted
  incrementally and reused in a FAIL lane instead of being lost.

Progress/evidence files contain only fixed-vocabulary categories, counts,
scenario ids/timings and exact SHA material. No prompts, user messages,
answers, passages or secrets are written here.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aa.qualification.product_contract_live import LaneResult

ENV_PROGRESS_PATH = "AA_QUALIFICATION_LIVE_PROGRESS_PATH"
ENV_TOTAL_DEADLINE_S = "AA_QUALIFICATION_LIVE_DEADLINE_S"
ENV_SCENARIO_GRACE_S = "AA_QUALIFICATION_WATCHDOG_SCENARIO_GRACE_S"
ENV_FINALISATION_RESERVE_S = "AA_QUALIFICATION_FINALISATION_RESERVE_S"
ENV_JOB_STARTED_AT = "AA_QUALIFICATION_JOB_STARTED_AT"
ENV_JOB_TIMEOUT_MINUTES = "AA_QUALIFICATION_JOB_TIMEOUT_MINUTES"
ENV_DISABLE_WATCHDOG = "AA_QUALIFICATION_WATCHDOG_DISABLED"

DEFAULT_TOTAL_DEADLINE_S = 5400.0
DEFAULT_SCENARIO_GRACE_S = 30.0
DEFAULT_FINALISATION_RESERVE_S = 900.0
DEFAULT_JOB_TIMEOUT_MINUTES = 120.0
DEFAULT_SCENARIO_WAIT_S = 300.0
DEFAULT_KILL_GRACE_S = 5.0
_POLL_INTERVAL_S = 0.2

PROGRESS_SCHEMA_VERSION = "aa-live-watchdog-progress/1"
_OUTCOME_SCHEMA_VERSION = "aa-live-watchdog-outcome/1"

_LaneRunner = Callable[..., Awaitable["LaneResult"]]


def watchdog_enabled() -> bool:
    """Whether the outer watchdog is active (enabled by default)."""
    raw = os.environ.get(ENV_DISABLE_WATCHDOG, "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not (value > 0.0) or value != value or value == float("inf"):
        return default
    return value


def scenario_grace_s() -> float:
    """Grace after the child's own scenario wait before parent escalation."""
    return _positive_float(ENV_SCENARIO_GRACE_S, DEFAULT_SCENARIO_GRACE_S)


def finalisation_reserve_s() -> float:
    """Time kept for Gate E/F, cleanup and artifact upload."""
    return _positive_float(ENV_FINALISATION_RESERVE_S, DEFAULT_FINALISATION_RESERVE_S)


def live_total_deadline_s(now: float | None = None) -> float:
    """Resolve the maximum live-lane duration in seconds.

    A configured ``AA_QUALIFICATION_LIVE_DEADLINE_S`` is authoritative but
    still capped by the remaining Actions job budget. Without a job-start
    marker the conservative repository default is used.
    """
    current = time.time() if now is None else now
    explicit = os.environ.get(ENV_TOTAL_DEADLINE_S, "").strip()
    total = (
        _positive_float(ENV_TOTAL_DEADLINE_S, DEFAULT_TOTAL_DEADLINE_S)
        if explicit
        else DEFAULT_TOTAL_DEADLINE_S
    )

    job_started_raw = os.environ.get(ENV_JOB_STARTED_AT, "").strip()
    if job_started_raw:
        try:
            job_started = float(job_started_raw)
        except (TypeError, ValueError):
            job_started = 0.0
        if job_started > 0.0:
            timeout_s = _positive_float(ENV_JOB_TIMEOUT_MINUTES, DEFAULT_JOB_TIMEOUT_MINUTES) * 60.0
            remaining = job_started + timeout_s - finalisation_reserve_s() - current
            if remaining <= 0.0:
                return 1.0
            # With a real Actions job-start marker the remaining budget is
            # authoritative; the static default is only the fallback for
            # direct/local invocation.
            total = min(total, remaining) if explicit else remaining
    return max(1.0, total)


def progress_path_from_env() -> Path | None:
    """Return the configured progress path, if any."""
    raw = os.environ.get(ENV_PROGRESS_PATH, "").strip()
    return Path(raw) if raw else None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_live_progress(path: Path | None = None) -> dict[str, Any]:
    """Read the last progress marker fail-soft (never raises)."""
    target = path or progress_path_from_env()
    if target is None:
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def record_live_progress(
    phase: str,
    *,
    scenario_index: int | None = None,
    family: str | None = None,
    scenario_deadline_at: float | None = None,
    clear_scenario_deadline: bool = False,
    completed_scenario_count: int | None = None,
    scenario_total: int | None = None,
    scenario_diagnostics: Sequence[Mapping[str, Any]] | None = None,
    latency_samples: Sequence[Mapping[str, Any]] | None = None,
    **extra: Any,
) -> None:
    """Persist one privacy-safe progress marker for the parent watchdog.

    Markers are merged so a heartbeat does not clear the active scenario
    deadline. Failures are swallowed: progress persistence must never break
    the actual qualification lane.
    """
    target = progress_path_from_env()
    if target is None:
        return
    try:
        payload = read_live_progress(target)
        payload.update(
            {
                "schema_version": PROGRESS_SCHEMA_VERSION,
                "pid": os.getpid(),
                "phase": str(phase)[:64],
                "updated_at": time.time(),
            }
        )
        if scenario_index is not None:
            payload["scenario_index"] = int(scenario_index)
        if family is not None:
            payload["family"] = str(family)[:64]
        if scenario_deadline_at is not None:
            payload["scenario_deadline_at"] = float(scenario_deadline_at)
        if clear_scenario_deadline:
            payload.pop("scenario_deadline_at", None)
        if completed_scenario_count is not None:
            payload["completed_scenario_count"] = int(completed_scenario_count)
        if scenario_total is not None:
            payload["scenario_total"] = int(scenario_total)
        if scenario_diagnostics is not None:
            payload["scenario_diagnostics"] = [dict(item) for item in scenario_diagnostics]
        if latency_samples is not None:
            payload["latency_samples"] = [dict(item) for item in latency_samples]
        payload.update(extra)
        _atomic_write_json(target, payload)
    except Exception:
        return


async def _await_lane(lane_runner: _LaneRunner, run_id: str) -> LaneResult:
    return await lane_runner(run_id=run_id)


def _child_main(
    lane_runner: _LaneRunner,
    run_id: str,
    outcome_path: str,
) -> None:
    """Child entrypoint: isolate the live lane in its own process group."""
    try:
        os.setsid()
    except OSError:
        pass
    outcome: dict[str, Any] = {"schema_version": _OUTCOME_SCHEMA_VERSION}
    try:
        import asyncio

        from aa.opencode.errors import OpenCodeRateLimitError

        try:
            result = asyncio.run(_await_lane(lane_runner, run_id))
        except OpenCodeRateLimitError as exc:
            outcome.update(
                {
                    "kind": "rate_limit",
                    "exception": type(exc).__name__,
                    "detail": str(exc)[:160],
                }
            )
        else:
            outcome.update({"kind": "result", "lane": result.to_dict()})
    except BaseException as exc:  # noqa: BLE001 - type-only cross-process evidence
        outcome.update({"kind": "error", "exception": type(exc).__name__})
    try:
        _atomic_write_json(Path(outcome_path), outcome)
    except Exception:
        pass


def _terminate_process_group(proc: Any, grace_s: float = DEFAULT_KILL_GRACE_S) -> None:
    """SIGTERM the child process group, then SIGKILL after a bounded grace."""
    pgid = proc.pid
    if pgid is None:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    proc.join(grace_s)
    if proc.is_alive():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.join(grace_s)
    # Best-effort final sweep for descendants that ignored SIGTERM.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _lane_from_payload(payload: Mapping[str, Any]) -> LaneResult:
    from aa.qualification.product_contract_live import LaneResult

    return LaneResult(
        lane=str(payload.get("lane", "live-telegram-evidence")),
        status=str(payload.get("status", "FAIL")),
        passed=tuple(str(item) for item in payload.get("passed", []) or []),
        failed=tuple(str(item) for item in payload.get("failed", []) or []),
        incomplete=tuple(str(item) for item in payload.get("incomplete", []) or []),
        metrics=dict(payload.get("metrics", {}) or {}),
    )


def _failed_lane(
    reason: str, *, main_sha: str = "", progress: Mapping[str, Any] | None = None
) -> LaneResult:
    from aa.qualification.product_contract_live import LaneResult

    metrics: dict[str, Any] = {
        "scenarios_executed": 0,
        "turns_executed": 0,
        "watchdog_triggered": False,
        "watchdog_reason": reason,
        "exact_sha": main_sha,
        "production_boundary": "PollingTelegramTransport.getUpdates->_process_raw_update",
    }
    if progress:
        metrics["watchdog_last_progress_phase"] = str(progress.get("phase", ""))[:64]
    return LaneResult(
        lane="live-telegram-evidence",
        status="FAIL",
        failed=(f"live-telegram-evidence:{reason}",),
        metrics=metrics,
    )


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _timeout_category(progress: Mapping[str, Any]) -> str:
    """Return a fixed-vocabulary timeout category from verified evidence."""
    recorded = str(progress.get("timeout_category", "") or "").strip()
    if recorded in {"stuck-provider", "stuck-retrieval", "unknown"}:
        return recorded
    stage = str(progress.get("active_stage", "") or "").strip()
    if stage in {"planner", "answer", "verifier", "opencode", "provider"}:
        return "stuck-provider"
    if stage in {"retrieval", "index"}:
        return "stuck-retrieval"
    return "unknown"


def _timeout_lane(
    *,
    trigger: str,
    progress: Mapping[str, Any],
    main_sha: str,
    elapsed_s: float,
    deadline_s: float,
) -> LaneResult:
    """Build typed FAIL evidence from persisted markers, never fake samples."""
    from aa.qualification.product_contract_live import LaneResult

    category = _timeout_category(progress)
    diagnostics_raw = progress.get("scenario_diagnostics", [])
    diagnostics = (
        [dict(item) for item in diagnostics_raw if isinstance(item, Mapping)]
        if isinstance(diagnostics_raw, list)
        else []
    )
    samples_raw = progress.get("latency_samples", [])
    samples = (
        [dict(item) for item in samples_raw if isinstance(item, Mapping)]
        if isinstance(samples_raw, list)
        else []
    )
    completed = _safe_int(progress.get("completed_scenario_count"), len(diagnostics))
    scenario_total = _safe_int(progress.get("scenario_total"), 18)
    last_index = _safe_int(progress.get("scenario_index"), 0)
    family = str(progress.get("family", "unknown"))[:64]
    timeout_entry = {
        "scenario_index": last_index,
        "family": family,
        "delivery_timeout": True,
        "watchdog_timeout": True,
        "timeout_category": category,
        "watchdog_trigger": trigger,
        "completed_scenario_count": completed,
        "scenario_total": scenario_total,
        "elapsed_ms": round(elapsed_s * 1000.0, 1),
    }
    if not any(isinstance(item, Mapping) and item.get("watchdog_timeout") for item in diagnostics):
        diagnostics.append(timeout_entry)
    turn_latencies_ms = [
        round(float(item.get("latency_ms", 0.0)), 1)
        for item in samples
        if isinstance(item.get("latency_ms"), (int, float))
    ]
    metrics: dict[str, Any] = {
        "scenarios_executed": completed,
        "scenario_total": scenario_total,
        "completed_scenario_count": completed,
        "last_scenario_index": last_index,
        "last_scenario_family": family,
        "scenario_timeout": True,
        "timeout_category": category,
        "watchdog_triggered": True,
        "watchdog_trigger": trigger,
        "watchdog_elapsed_ms": round(elapsed_s * 1000.0, 1),
        "watchdog_deadline_s": round(deadline_s, 3),
        "watchdog_last_progress_phase": str(progress.get("phase", ""))[:64],
        "watchdog_last_progress_at": progress.get("updated_at"),
        "exact_sha": main_sha,
        "scenario_diagnostics": diagnostics,
        "turn_latency_samples": samples,
        "turn_latencies_ms": turn_latencies_ms,
        "turns_executed": len(samples),
        "qualification_scenario_wait_s": _positive_float(
            "AA_QUALIFICATION_SCENARIO_WAIT_S", DEFAULT_SCENARIO_WAIT_S
        ),
        "live_prerequisites_present": False,
        "production_boundary": "PollingTelegramTransport.getUpdates->_process_raw_update",
        "gate_boundary": {
            "real_provider": True,
            "real_book": True,
            "real_app": True,
            "telegram_ingress_mode": "production-adapter",
            "telegram_egress_mode": "stubbed",
            "delivery_receipt_kind": "simulated-in-process",
            "network_probe_consent": "absent",
        },
        "telegram_egress_label": "SIMULATED",
        "external_send_status": "external_send_unverified",
    }
    return LaneResult(
        lane="live-telegram-evidence",
        status="FAIL",
        failed=(f"live-scenario-watchdog-timeout-{category}",),
        metrics=metrics,
    )


def supervise_live_gate_c(
    lane_runner: _LaneRunner,
    *,
    run_id: str,
    main_sha: str = "",
    progress_path: Path | None = None,
) -> LaneResult:
    """Run the live Gate C lane under the independent outer watchdog."""
    from aa.opencode.errors import OpenCodeRateLimitError

    target_progress = progress_path or progress_path_from_env()
    if target_progress is None:
        target_progress = Path(tempfile.gettempdir()) / f"aa-live-watchdog-{os.getpid()}.json"
    os.environ[ENV_PROGRESS_PATH] = str(target_progress)
    try:
        target_progress.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass

    outcome_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="aa-live-watchdog-outcome-", suffix=".json", delete=False
    )
    outcome_path = Path(outcome_file.name)
    outcome_file.close()

    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(
        target=_child_main,
        args=(lane_runner, run_id, str(outcome_path)),
        daemon=True,
    )
    started_monotonic = time.monotonic()
    total_deadline_s = live_total_deadline_s()
    deadline_monotonic = started_monotonic + total_deadline_s
    trigger = ""
    last_progress: dict[str, Any] = {}
    proc.start()
    try:
        while True:
            proc.join(_POLL_INTERVAL_S)
            progress = read_live_progress(target_progress)
            if progress:
                last_progress = progress
            if not proc.is_alive():
                break
            now_monotonic = time.monotonic()
            if now_monotonic >= deadline_monotonic:
                trigger = "total-deadline"
                break
            scenario_deadline = last_progress.get("scenario_deadline_at")
            if isinstance(scenario_deadline, (int, float)):
                if time.time() >= float(scenario_deadline):
                    trigger = "scenario-deadline"
                    break
        if trigger:
            elapsed = time.monotonic() - started_monotonic
            _terminate_process_group(proc)
            last_progress = read_live_progress(target_progress) or last_progress
            try:
                outcome_path.unlink()
            except OSError:
                pass
            return _timeout_lane(
                trigger=trigger,
                progress=last_progress,
                main_sha=main_sha,
                elapsed_s=elapsed,
                deadline_s=total_deadline_s,
            )
        proc.join()
    finally:
        if proc.is_alive():
            _terminate_process_group(proc)

    try:
        try:
            payload = json.loads(outcome_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _failed_lane(
                "watchdog-child-no-outcome", main_sha=main_sha, progress=last_progress
            )
        if not isinstance(payload, dict):
            return _failed_lane(
                "watchdog-child-invalid-outcome", main_sha=main_sha, progress=last_progress
            )
        kind = str(payload.get("kind", ""))
        if kind == "rate_limit":
            raise OpenCodeRateLimitError(
                str(payload.get("detail", "opencode request rate-limited"))
            )
        if kind == "error":
            exception = str(payload.get("exception", "UnknownError"))[:96]
            return _failed_lane(exception, main_sha=main_sha, progress=last_progress)
        lane_payload = payload.get("lane")
        if kind == "result" and isinstance(lane_payload, Mapping):
            return _lane_from_payload(lane_payload)
        return _failed_lane(
            "watchdog-child-invalid-outcome", main_sha=main_sha, progress=last_progress
        )
    finally:
        try:
            outcome_path.unlink()
        except OSError:
            pass


__all__ = [
    "DEFAULT_FINALISATION_RESERVE_S",
    "DEFAULT_SCENARIO_GRACE_S",
    "DEFAULT_TOTAL_DEADLINE_S",
    "ENV_DISABLE_WATCHDOG",
    "ENV_FINALISATION_RESERVE_S",
    "ENV_JOB_STARTED_AT",
    "ENV_JOB_TIMEOUT_MINUTES",
    "ENV_PROGRESS_PATH",
    "ENV_SCENARIO_GRACE_S",
    "ENV_TOTAL_DEADLINE_S",
    "finalisation_reserve_s",
    "live_total_deadline_s",
    "progress_path_from_env",
    "read_live_progress",
    "record_live_progress",
    "scenario_grace_s",
    "supervise_live_gate_c",
    "watchdog_enabled",
]
