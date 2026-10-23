"""Independent stall-snapshot observer sidecar (kodmial/aa#431).

The previous diagnostics lost per-attempt evidence exactly when a request
hung: the request latency was recorded only after the call ended, so a
stuck ``urllib.read()`` produced nothing. Fixed snapshots are taken by an
INDEPENDENT monitor here at 10/25/45/75/110 seconds into every
non-responding attempt, persisted immediately to disk, and never depend on
the state of the wedged worker thread. Traces persist on normal
completion, failure, cancellation, job finish, and even at interpreter
exit.

Snapshot/trace payloads are strictly privacy-safe: attempt identity,
operation category, agent/model pointers, bounded runtime event filters,
bounded session-status probe verdicts — never prompts, replies, book
passages, secrets, Telegram/chat/session identifiers, or raw OpenCode
stdout/log lines.

Typing heartbeats, SSE events and health probes are NOT provider token
progress; nothing here upgrades them to progress evidence.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

OBSERVER_INTERVAL_S = 0.5
SNAPSHOT_DELAYS_S: tuple[float, ...] = (10.0, 25.0, 45.0, 75.0, 110.0)
SCHEMA_STALL_SNAPSHOT = "aa-opencode-stall-snapshot/1"
SCHEMA_STALL_ATTEMPT_TRACE = "aa-opencode-stall-attempt-trace/1"
SCHEMA_STALL_FLUSH = "aa-opencode-stall-flush/1"

# Allow-listed attempt fields copied into privacy-safe snapshots.
_ATTEMPT_EXPORT_FIELDS = (
    "attempt_id",
    "operation",
    "state",
    "causality_id",
    "stage",
    "agent",
    "audit_agent",
    "model",
    "fallback_model",
    "attempt_started_monotonic",
    "attempt_ended_monotonic",
    "latency_ms",
    "failure_category",
    "error_kind",
    "late_outcome",
    "late_outcome_source",
    "late_outcome_conflicts",
    "late_failure_category",
    "late_latency_ms",
    "worker_done",
    "clean_worker_cancel",
)

_MAX_TRACE_FILES_PER_ATTEMPT = 1
_MAX_BYTES_PER_SNAPSHOT_BODY = 64


def attempt_view(entry: Mapping[str, object]) -> dict[str, Any]:
    """Privacy-safe bounded projection of one attempt ledger row."""
    return {
        str(field): entry.get(field)
        for field in _ATTEMPT_EXPORT_FIELDS
        if entry.get(field) is not None
    }


_attempt_view = attempt_view


class StallSnapshotObserver:
    """Snapshot every non-responding OpenCode attempt at fixed delays.

    ``attempt_source`` supplies the current privacy-safe attempt records
    (the OpenCode client's attempt ledger). ``process_state_source``
    optionally supplies a bounded runtime descriptor (alive/exit code) and
    ``session_status_probe`` an optional async direct session probe; probe
    results are reduced to bounded verdicts before persistence, and the
    probe never receives prompt text.
    """

    def __init__(
        self,
        *,
        attempt_source: Callable[[], Sequence[Mapping[str, Any]]],
        out_dir: Path | str,
        delays: Sequence[float] = SNAPSHOT_DELAYS_S,
        interval_s: float = OBSERVER_INTERVAL_S,
        process_state_source: Callable[[], Mapping[str, Any]] | None = None,
        session_status_probe: Callable[[], Awaitable[Mapping[str, Any]]] | None = None,
        register_atexit_flush: bool = False,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if not delays or any(delay <= 0 for delay in delays):
            raise ValueError("delays must be a non-empty sequence of positive seconds")
        self._attempt_source = attempt_source
        self._out_dir = Path(out_dir) if not isinstance(out_dir, Path) else out_dir
        self._delays = tuple(sorted(set(float(delay) for delay in delays)))
        self._interval_s = float(interval_s)
        self._process_state_source = process_state_source
        self._session_status_probe = session_status_probe
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._snapshot_attempts: dict[tuple[str, float], str] = {}
        # Insertion-ordered so eviction drops the OLDEST trace ids, not the
        # alphabetically smallest ones.
        self._trace_written: dict[str, None] = {}
        self._registered_atexit = False
        self._atexit_registered = False
        if register_atexit_flush:
            import atexit

            atexit.register(self._atexit_flush)
            self._atexit_registered = True

    @property
    def out_dir(self) -> Path:
        """The privacy-safe snapshot persistence directory."""
        return self._out_dir

    def _atexit_flush(self) -> None:
        """Interpreter-exit hook: flush the index without loop involvement."""
        self.flush("interpreter-exit", sync=True)

    def _snapshot_path(self, attempt_id: str, delay: float) -> Path:
        safe_id = str(attempt_id)[:64].replace("/", "_") or "unknown"
        return self._out_dir / f"attempt-{safe_id}-snapshot-{int(delay)}s.json"

    def _trace_path(self, attempt_id: str) -> Path:
        safe_id = str(attempt_id)[:64].replace("/", "_") or "unknown"
        return self._out_dir / f"attempt-{safe_id}-trace.json"

    def _index_path(self) -> Path:
        return self._out_dir / "attempt-observer-index.json"

    def _process_state(self) -> dict[str, Any]:
        if self._process_state_source is None:
            return {}
        try:
            payload = self._process_state_source()
        except Exception:
            return {"process_state_error": "unavailable"}
        if not isinstance(payload, Mapping):
            return {}
        bounded: dict[str, Any] = {}
        for key in ("alive", "exit_code", "spawned", "started"):
            if key in payload:
                bounded[str(key)[:32]] = payload[key]
        return bounded

    async def _probe_session_status(self) -> dict[str, Any]:
        if self._session_status_probe is None:
            return {}
        try:
            payload = await self._session_status_probe()
        except Exception as exc:
            # Never store chat/session ids or provider error bodies here:
            # only the bounded observed verdict and an exception type name.
            return {
                "session_status_observed": False,
                "session_status_error": type(exc).__name__[:64],
                "progress_meaning": "none",
            }
        if not isinstance(payload, Mapping):
            return {
                "session_status_observed": False,
                "session_status_error": "invalid-probe-payload",
                "progress_meaning": "none",
            }
        observed = bool(payload.get("observed", True))
        return {
            "session_status_observed": observed,
            "session_status_verdict": str(payload.get("verdict", ""))[:64],
            "session_status_error": str(payload.get("error", ""))[:64] if not observed else "",
            # Deliberate honesty: this probe is not provider token progress.
            "progress_meaning": "none",
        }

    async def snapshot_once(self, entry: Mapping[str, Any], *, delay: float) -> Path:
        """Persist one bounded snapshot for a non-responding attempt."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        attempt_id = str(entry.get("attempt_id", ""))
        started = entry.get("attempt_started_monotonic")
        elapsed = (
            max(0.0, time.monotonic() - float(started))
            if isinstance(started, (int, float))
            else 0.0
        )
        snapshot: dict[str, Any] = {
            "schema": SCHEMA_STALL_SNAPSHOT,
            "delay_s": delay,
            "captured_at_s": round(time.time(), 1),
            "elapsed_s": round(elapsed, 1),
            "attempt": _attempt_view(entry),
            "process": self._process_state(),
            "session_status": await self._probe_session_status(),
            "uncertainty": (
                "session-status/health/typing probes do not prove provider token progress; "
                "the attempt ledger is the only direct request evidence"
            ),
        }
        path = self._snapshot_path(attempt_id, delay)
        _write_json(path, snapshot)
        return path

    async def trace_attempt(self, entry: Mapping[str, Any]) -> Path:
        """Persist the terminal trace of one attempt (completion/failure/cancel)."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        attempt_id = str(entry.get("attempt_id", ""))
        trace: dict[str, Any] = {
            "schema": SCHEMA_STALL_ATTEMPT_TRACE,
            "captured_at_s": round(time.time(), 1),
            "attempt": _attempt_view(entry),
            "process": self._process_state(),
            "snapshots": sorted(
                marker for key, marker in self._snapshot_attempts.items() if key[0] == attempt_id
            ),
        }
        path = self._trace_path(attempt_id)
        _write_json(path, trace)
        return path

    def snapshot_paths_for(self, attempt_id: str) -> list[str]:
        return sorted(
            marker for key, marker in self._snapshot_attempts.items() if key[0] == attempt_id
        )

    def flush(self, reason: str, *, sync: bool = False) -> Path:
        """Persist the observer index; runs even when a child is stuck.

        Written on normal completion, failure, cancellation, job finish,
        and crash (optionally via the interpreter-exit hook).
        """
        self._out_dir.mkdir(parents=True, exist_ok=True)
        index: dict[str, Any] = {
            "schema": SCHEMA_STALL_FLUSH,
            "reason": str(reason)[:64],
            "flushed_at_s": round(time.time(), 1),
            "snapshot_attempts": sum(1 for _ in self._snapshot_attempts),
            "traced_attempts": len(self._trace_written),
            "delays_s": list(self._delays),
        }
        path = self._index_path()
        # The index is live progress evidence: external consumers must see
        # new snapshots/traces without waiting for the observer to stop.
        _write_json(path, index)
        return path

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = asyncio.Event()
            self._task = asyncio.create_task(self._run(), name="aa-stall-snapshot-observer")

    async def stop(self, *, flush_reason: str = "observer-stopped") -> None:
        self._stopping.set()
        task = self._task
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
        self._task = None
        # Final bounded scan (kodmial/aa#431): a terminal transition that
        # lands between the last interval scan and stop must still persist
        # its per-attempt trace before the index flush.
        try:
            await asyncio.wait_for(self._scan_once(), timeout=1.0)
        except (TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        self.flush(flush_reason)

    async def _run(self) -> None:
        while not self._stopping.is_set():
            await self._scan_once()
            self.flush("observer-scan")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval_s)
            except TimeoutError:
                continue

    async def _scan_once(self) -> None:
        now = time.monotonic()
        # Take one defensive copy per entry before inspecting it. A source
        # may hand back rows that are still mutated in place by the writer;
        # copying once keeps all fields of a scan consistent with each other.
        snapshots: list[dict[str, Any]] = []
        for entry in self._attempt_source():
            if isinstance(entry, Mapping):
                snapshots.append(dict(entry))
        for entry in snapshots:
            attempt_id = str(entry.get("attempt_id", ""))
            if not attempt_id:
                continue
            started = entry.get("attempt_started_monotonic")
            elapsed = max(0.0, now - float(started)) if isinstance(started, (int, float)) else 0.0
            if entry.get("state") == "in-flight":
                for delay in self._delays:
                    key = (attempt_id, delay)
                    if key in self._snapshot_attempts:
                        continue
                    if elapsed >= delay:
                        path = await self.snapshot_once(entry, delay=delay)
                        self._snapshot_attempts[key] = str(path)
            elif entry.get("state") in ("completed", "failed", "cancelled", "aborted"):
                if attempt_id in self._trace_written:
                    continue
                path = await self.trace_attempt(entry)
                self._trace_written[attempt_id] = None
                if len(self._trace_written) >= 1024:
                    stale = list(self._trace_written)[: len(self._trace_written) - 512]
                    for old_id in stale:
                        del self._trace_written[old_id]
                _ = path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > 1 << 20:
        raise ValueError("stall snapshot payload exceeded the bounded size")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


__all__ = [
    "OBSERVER_INTERVAL_S",
    "SCHEMA_STALL_ATTEMPT_TRACE",
    "SCHEMA_STALL_FLUSH",
    "SCHEMA_STALL_SNAPSHOT",
    "SNAPSHOT_DELAYS_S",
    "StallSnapshotObserver",
    "attempt_view",
]
