#!/usr/bin/env python3
"""Gate C scenario #2 focused diagnostic (issue #418) — P0 isolated probe.

DIAGNOSTIC ONLY. This script never decides Gate A-F qualification, never
touches the product criteria of issue #7 and never reruns the full 18-case
Gate C. It runs exactly two fast, separable probes against the real
production runtime on the exact checked-out SHA:

- Probe A (``direct-model``): real local ``opencode serve`` (spawned by the
  production ``LocalOpenCodeRuntime``), the approved AA model pointer, one
  simple session + one short message. Per-phase HTTP timing (connect /
  write / read) is captured so a direct model/provider stall is proven
  before anything else is blamed;
- Probe B (``scenario2``): the exact production scenario #2
  (``substantive-drinking``) through the unchanged production boundary
  (``PollingTelegramTransport`` -> ``Application`` -> per-chat dispatcher ->
  LangGraph planner/retrieval/answer/verifier), using the deterministic
  in-process Telegram ``getUpdates``/``sendMessage`` seam (NO external
  Telegram network) and the unchanged scenario #2 prompt.

Privacy: no user text, no book text, no model payloads, no tokens, no
session ids, no URLs with credentials and no raw environment is ever
printed or written. Events carry ids, categories, counts, latencies and
code identifiers only. The scenario prompt itself is never logged.

Progressive evidence: line-buffered timestamped JSONL stage START/END,
provider lifecycle and last-real-progress events stream to stdout (the
GitHub Actions log) and to privacy-safe artifact files. An in-child
watchdog thread plus the parent supervisor take snapshots at fixed offsets
(default 10/25/45/75/110 s of the target turn); each snapshot also writes
a small privacy-safe checkpoint JSON. The workflow additionally updates
exactly one diagnostic issue comment, idempotent per run.

Diagnostic cutoff: after ``--diagnostic-cutoff-s`` (default 130 s) without
a delivered turn, the supervisor captures one final stack + process +
socket + server snapshot and then SIGTERM/SIGKILLs the child's whole
process group. A slow but genuinely progressing turn is reported
``still-in-progress-at-probe-end`` (INCONCLUSIVE) — this observation
window is NOT a new product qualification timeout.

Usage (workflow):
  python scripts/diagnose_gate_c_scenario2.py \
    --main-sha <sha> --run-id <run_id> --issue 418 --out-dir diagnostic-scenario2-out
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Hard-coded verified owner target (kodmial/aa#418); the workflow pins it too.
DEFAULT_OWNER = "kodmial"
DEFAULT_REPO = "aa"
DEFAULT_ISSUE = 418

SCHEMA_VERSION = "aa-gate-c-scenario2-diagnostic/1"
PROBE_DIRECT_MODEL = "direct-model"
PROBE_SCENARIO2 = "scenario2"
SCENARIO_2_FAMILY = "substantive-drinking"
SCENARIO_2_CHAT_ID = 920002
# Unchanged scenario #2 core prompt (product_contract_live core_scenarios[1]).
# Privacy: never logged; travels only through the production update seam.
SCENARIO_2_PROMPT = "К вечеру очень тянет выпить, как с этим обходиться?"

DEFAULT_SNAPSHOTS = (10.0, 25.0, 45.0, 75.0, 110.0)
DEFAULT_DIAGNOSTIC_CUTOFF_S = 130.0
DEFAULT_PROBE_A_SNAPSHOTS = (5.0, 15.0, 30.0, 50.0, 70.0)
DEFAULT_PROBE_A_CUTOFF_S = 75.0
DEFAULT_PROBE_A_MESSAGE_TIMEOUT_S = 45.0
DEFAULT_PROBE_A_PORT = "4097"
CHILD_CUTOFF_SLACK_S = 10.0
SUPERVISOR_POLL_S = 0.25
STACK_DUMP_EVERY_S = 20.0
MAX_STACK_DUMPS = 8
COMMENT_CADENCE_S = 30.0
DEFAULT_TERMINATE_GRACE_S = 5.0

# Internal child/supervisor contract only (diagnostic, never a Gate verdict):
# 0 evidence produced, 2 bootstrap failed, 3 still-in-progress at cutoff,
# 4 report could not be produced, 5 unexpected child exception.
EXIT_OK = 0
EXIT_BOOTSTRAP_FAILED = 2
EXIT_STILL_IN_PROGRESS = 3
EXIT_REPORT_FAILED = 4
EXIT_CHILD_UNEXPECTED = 5

# Event kinds marking confirmed REAL progress; typing/polling heartbeats are
# deliberately excluded, mirroring the live watchdog semantics.
REAL_ACTIVITY_KINDS = frozenset(
    {
        "probe-start",
        "server-health",
        "session-create-start",
        "session-create-end",
        "model-send-start",
        "model-send-end",
        "bootstrap-start",
        "bootstrap-end",
        "stage-start",
        "delivery",
        "scenario-start",
        "scenario-end",
    }
)

# Observed production semantic-DAG op names -> root-cause component.
_STAGE_COMPONENT: dict[str, str] = {
    "opencode": "model-provider",
    "provider": "model-provider",
    "planner": "aa-graph-planner",
    "index": "aa-retrieval-index",
    "retrieval": "aa-retrieval",
    "answer": "aa-graph-answer",
    "verifier": "aa-graph-verifier",
    "whole_turn_judge": "aa-graph-judge",
    "finalize": "aa-graph-finalization",
    "turn": "aa-graph-orchestration",
}

# Privacy: keys that must never carry any text in diagnostic payloads.
_FORBIDDEN_KEYS = frozenset(
    {
        "text",
        "exact_text",
        "utterance",
        "answer",
        "generated_answer",
        "evidence_text",
        "source_text",
        "passage_text",
        "book_text",
        "content",
        "synthetic_input",
        "transcript",
        "summary",
        "prompt",
        "chain_of_thought",
        "secret",
        "identity",
        "private_key",
        "token",
        "session_id",
        "url",
    }
)


def assert_no_text_leak(payload: Any, owner: str = "scenario2-diagnostic") -> None:
    """Reject text-carrying forbidden keys in diagnostic payloads (fail-closed)."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = key.decode("utf-8", "replace") if isinstance(key, bytes) else str(key)
            if key_text.lower() in _FORBIDDEN_KEYS and isinstance(value, str) and value:
                raise ValueError(f"{owner}: forbidden key {key_text!r} carries text")
            assert_no_text_leak(value, f"{owner}.{key_text}")
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            assert_no_text_leak(item, f"{owner}[{index}]")


def now_iso() -> str:
    """UTC timestamp (wall clock) for diagnostic events."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


class EventSink:
    """Line-buffered timestamped JSONL emitter: artifact file + stdout.

    Every line is flushed immediately so evidence exists in the Actions
    log long before any terminal state.
    """

    def __init__(
        self, path: Path, *, echo: bool = True, process_start: float | None = None
    ) -> None:
        self.echo = echo
        self._start = process_start if process_start is not None else time.monotonic()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Emit one JSONL event (file + stdout, flushed)."""
        payload: dict[str, Any] = {
            "ts": now_iso(),
            "mono": round(time.monotonic() - self._start, 3),
            "event": str(event),
        }
        payload.update(fields)
        assert_no_text_leak({event: payload})
        line = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        self._file.write(line + "\n")
        if self.echo:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        return payload

    def passthrough(self, line: str) -> None:
        """Relay one child stdout line verbatim into the Actions log."""
        if not self.echo:
            return
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


def parse_schedule(raw: str) -> tuple[float, ...]:
    """Parse "10,25,45,75,110" into ascending positive floats."""
    values: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = float(chunk)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return tuple(sorted(values))


def sanitize_frame(code: Any) -> str:
    """Code identifier summary for one frame (never any runtime text data)."""
    try:
        file_name = str(Path(str(code.co_filename)).name)
        qualifier = getattr(code, "co_qualname", code.co_name)
        return f"{file_name}:{qualifier}"
    except Exception:
        return "<opaque-frame>"


def summarize_loopback_sockets() -> dict[str, int]:
    """Loopback socket summary by state/port from /proc (privacy-safe)."""
    counts: dict[str, int] = {
        "total": 0,
        "established_to_4096": 0,
        "established_to_4097": 0,
        "established_other": 0,
        "listen_4096": 0,
        "listen_4097": 0,
        "other_states": 0,
    }
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            counts["total"] += 1
            state = parts[3]
            try:
                local_port = int(parts[1].rsplit(":", 1)[1], 16)
                remote_port = int(parts[2].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if state == "0A":
                if local_port == 4096:
                    counts["listen_4096"] += 1
                elif local_port == 4097:
                    counts["listen_4097"] += 1
                else:
                    counts["other_states"] += 1
            elif state == "01":
                if 4096 in (local_port, remote_port):
                    counts["established_to_4096"] += 1
                elif 4097 in (local_port, remote_port):
                    counts["established_to_4097"] += 1
                else:
                    counts["established_other"] += 1
            else:
                counts["other_states"] += 1
    return counts


def count_opencode_processes() -> int:
    """Count processes named ``opencode`` (no command lines, no user names)."""
    count = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            comm = Path(f"/proc/{entry}/comm").read_text(encoding="utf-8").strip()
            if comm == "opencode":
                count += 1
        except OSError:
            continue
    return count


def http_split_probe_base_url(base_url: str) -> tuple[str, int]:
    """Split a validated loopback base URL (host, port)."""
    cleaned = base_url.strip().rstrip("/")
    if not cleaned.startswith("http://") or "127.0.0.1" not in cleaned:
        raise ValueError("diagnostic base URL must be loopback http")
    hostname, _, tail = cleaned[len("http://") :].partition(":")
    port = int(tail.split("/", 1)[0])
    return hostname, port


def _install_stack_handler(path: Path) -> None:
    """Turn SIGUSR1 into a whole-process faulthandler dump (append file)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a", encoding="utf-8")
        faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True, chain=True)
        _HELD_STACK_STREAMS.append(stream)
    except (OSError, ValueError):
        pass


_HELD_STACK_STREAMS: list[Any] = []


class ChildWatchdog(threading.Thread):
    """Independent in-child snapshot thread (survives event-loop wedges).

    Fires at each scheduled offset relative to the phase anchor (the target
    turn start). Emits JSONL ``snapshot`` events and writes one privacy-safe
    checkpoint JSON per snapshot. Whole-process faulthandler stacks are
    separately requested by the supervisor via SIGUSR1.
    """

    _MAX_THREADS = 12
    _MAX_FRAMES = 8
    _MAX_TASKS = 12

    def __init__(
        self,
        *,
        probe: str,
        schedule: tuple[float, ...],
        anchor: threading.Event,
        anchor_wait_timeout_s: float,
        sink: EventSink,
        checkpoint_dir: Path,
        opencode_base_url: str,
        counters: dict[str, Any],
        audit_provider: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__(name=f"scenario2-diag-watchdog-{probe}", daemon=True)
        self.probe = probe
        self.schedule = sorted(float(value) for value in schedule)
        self.anchor = anchor
        self.anchor_wait_timeout_s = max(1.0, float(anchor_wait_timeout_s))
        self.sink = sink
        self.checkpoint_dir = checkpoint_dir
        self.opencode_base_url = opencode_base_url
        self.counters = counters
        self.audit_provider = audit_provider
        self.main_loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()

    def run(self) -> None:
        first_offset = self.schedule[0] if self.schedule else self.anchor_wait_timeout_s
        if not self.anchor.wait(timeout=max(first_offset, 1.0) + self.anchor_wait_timeout_s):
            return
        start = time.monotonic()
        for index, offset in enumerate(self.schedule, start=1):
            while True:
                if self._stop.is_set():
                    return
                remaining = offset - (time.monotonic() - start)
                if remaining <= 0:
                    break
                self._stop.wait(timeout=min(0.2, remaining))
            if self._stop.is_set():
                return
            self._snapshot(index=index, at_s=round(time.monotonic() - start, 2))

    def stop(self) -> None:
        self._stop.set()

    def _snapshot(self, *, index: int, at_s: float) -> None:
        payload = {"type": "snapshot", "snapshot_index": index, **self._collect(at_s=at_s)}
        try:
            checkpoint = self.checkpoint_dir / f"checkpoint-{self.probe}-{index:03d}.json"
            checkpoint.write_text(
                json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        try:
            self.sink.emit("snapshot", **{k: v for k, v in payload.items() if k != "type"})
        except Exception:
            pass

    def _collect(self, *, at_s: float) -> dict[str, Any]:
        threads: list[dict[str, Any]] = []
        frame_map = sys._current_frames()
        for thread in threading.enumerate()[-self._MAX_THREADS :]:
            frames: list[str] = []
            frame = frame_map.get(thread.ident)
            for _ in range(self._MAX_FRAMES):
                if frame is None:
                    break
                frames.append(sanitize_frame(frame.f_code))
                frame = frame.f_back
            threads.append({"name": thread.name[:48], "frames": frames})
        tasks: list[dict[str, Any]] = []
        loop = self.main_loop
        if loop is not None and not loop.is_closed():
            try:
                pending = list(asyncio.all_tasks(loop))[: self._MAX_TASKS]
            except RuntimeError:
                pending = []
            for task in pending:
                try:
                    label = sanitize_frame(task.get_coro().cr_code)
                except Exception:
                    label = "<opaque-task>"
                tasks.append({"name": task.get_name()[:48], "coro": label, "done": task.done()})
        health_ok: Any = None
        health_elapsed_ms: Any = None
        try:
            probe_start = time.monotonic()
            health_ok, _ = _health_probe(self.opencode_base_url, 2.0)
            health_elapsed_ms = round((time.monotonic() - probe_start) * 1000.0, 1)
        except Exception:
            health_ok = None
        return {
            "probe": self.probe,
            "at_s": at_s,
            "stage": str(self.counters.get("last_stage", "") or "")[:64],
            "stage_at_s": self.counters.get("last_stage_at_s"),
            "seconds_since_last_stage": self.counters.get("seconds_since_last_stage"),
            "last_real_event": str(self.counters.get("last_real_event", "") or "")[:64],
            "seconds_since_last_real": self.counters.get("seconds_since_last_real"),
            "typing_sends": int(self.counters.get("typing_sends", 0)),
            "delivered": bool(self.counters.get("delivered", False)),
            "threads": threads,
            "asyncio_tasks": tasks,
            "opencode_health": {"ok": health_ok, "elapsed_ms": health_elapsed_ms},
            "opencode_request_audit": self._audit_summary(),
            "sockets": summarize_loopback_sockets(),
            "opencode_processes": count_opencode_processes(),
        }

    def _audit_summary(self) -> dict[str, Any]:
        """Privacy-safe OpenCode client operation audit (counts + latency)."""
        summary: dict[str, Any] = {"present": False}
        provider = self.audit_provider
        if provider is None:
            return summary
        try:
            audit = list(provider())[-12:]
            summary["present"] = bool(audit)
            summary["entries"] = [
                {
                    "operation": str(item.get("operation", ""))[:32],
                    "latency_ms": item.get("latency_ms"),
                    "success": bool(item.get("success")),
                }
                for item in audit
                if isinstance(item, dict)
            ]
        except Exception:
            summary["present"] = False
        return summary


def _health_probe(base_url: str, timeout: float) -> tuple[bool, str]:
    """One bounded health probe; returns ``(healthy, version)``."""
    try:
        connection = http.client.HTTPConnection(*_split_base(base_url), timeout=timeout)
    except (ValueError, OSError):
        return False, ""
    try:
        connection.request("GET", "/global/health", headers={"Accept": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return False, ""
    finally:
        try:
            connection.close()
        except Exception:
            pass
    healthy = bool(isinstance(payload, dict) and payload.get("healthy") is True)
    version = payload.get("version") if healthy else ""
    return healthy, version if isinstance(version, str) else ""


def _split_base(base_url: str) -> tuple[str, int]:
    """Split base URL into (host, port) without exposing it anywhere."""
    cleaned = base_url.strip().rstrip("/")
    if not cleaned.startswith("http://"):
        raise ValueError("base URL must be http")
    hostname, _, tail = cleaned[len("http://") :].partition(":")
    port = int(tail.split("/", 1)[0])
    return hostname, port


def probe_health(base_url: str, timeout: float) -> tuple[bool, str]:
    """Public alias so the worker module owns one health implementation."""
    return _health_probe(base_url, timeout)


def _health_probe_phased(base_url: str, method: str, path: str) -> dict[str, float]:
    """One health call with per-phase timing (diagnostic-only helper)."""
    started_connect = time.perf_counter()
    connection = http.client.HTTPConnection(*_split_base(base_url), timeout=2.0)
    try:
        connection.connect()
        connect_ms = round((time.perf_counter() - started_connect) * 1000.0, 1)
        write_start = time.perf_counter()
        connection.request(method, path, headers={"Accept": "application/json"})
        write_ms = round((time.perf_counter() - write_start) * 1000.0, 1)
        read_start = time.perf_counter()
        response = connection.getresponse()
        response.read()
        read_ms = round((time.perf_counter() - read_start) * 1000.0, 1)
        return {"connect_ms": connect_ms, "write_ms": write_ms, "read_ms": read_ms}
    finally:
        try:
            connection.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Probe A (direct-model): real local opencode serve + approved model
# ---------------------------------------------------------------------------


def _timed_request(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, object] | None,
    timeout: float,
) -> tuple[bool, bool, dict[str, float], dict[str, Any]]:
    """One instrumented loopback HTTP call with connect/write/read phases.

    Diagnostic-only helper mirroring the exact production wire shape
    (stdlib ``http.client``). Response payloads are never returned or
    logged. The returned ``extra`` may hold an internal opaque ``_id``
    for the very next request; it is never emitted.
    """
    try:
        host, port = http_split_probe_base_url(base_url)
    except ValueError:
        return False, False, {"fail_category": "bad-base-url"}, {}
    timings: dict[str, float] = {}
    extra: dict[str, Any] = {}
    connection: http.client.HTTPConnection | None = None
    try:
        start = time.perf_counter()
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
        connection.connect()
        timings["connect_ms"] = round((time.perf_counter() - start) * 1000.0, 1)
        write_start = time.perf_counter()
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=payload, headers=headers)
        timings["write_ms"] = round((time.perf_counter() - write_start) * 1000.0, 1)
        read_start = time.perf_counter()
        response = connection.getresponse()
        raw = response.read()
        timings["read_ms"] = round((time.perf_counter() - read_start) * 1000.0, 1)
        timings["status"] = int(response.status)
        ok = response.status == 200
        if ok and method == "POST" and path == "/session":
            session = json.loads(raw.decode("utf-8"))
            session_id = session.get("id")
            if isinstance(session_id, str) and session_id:
                extra["_id"] = session_id
        if ok:
            decision = _classify_opencode_payload(raw)
            extra.update(decision)
            return ok, decision.get("provider_error", False), timings, extra
        extra.update({"fail_category": f"http-error-{timings.get('status', '?')}"})
        return ok, False, timings, extra
    except (TimeoutError, OSError) as exc:
        timings["fail_category"] = (
            "inactivity-timeout" if isinstance(exc, TimeoutError) else "network-error"
        )
        return False, False, timings, extra
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _classify_opencode_payload(raw: bytes) -> dict[str, Any]:
    """Classify one response without keeping any payload text."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {"ok": False, "provider_error": True}
    error = None
    served_model = ""
    if isinstance(payload, dict):
        info = payload.get("info")
        if isinstance(info, dict):
            error = info.get("error")
            served_model = _served_model(info)
    result: dict[str, Any] = {
        "ok": error in (None, False),
        "provider_error": error is not None,
        "served_model": served_model[:96],
    }
    if isinstance(error, dict):
        status_code = error.get("data", {}).get("statusCode")
        if isinstance(status_code, int):
            result["error_status_code"] = int(status_code)
    return result


def _served_model(info: dict[str, Any]) -> str:
    """Extract provider/model from OpenCode assistant metadata (safe)."""
    candidates: list[Any] = [info, info.get("assistant")]
    metadata = info.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend((metadata, metadata.get("assistant")))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        provider = candidate.get("providerID", candidate.get("provider_id", ""))
        model = candidate.get("modelID", candidate.get("model_id", ""))
        if isinstance(provider, str) and isinstance(model, str) and provider and model:
            return f"{provider}/{model}"
    return ""


def _probe_a_message_body(model: str) -> dict[str, object]:
    """Exact production wire shape for one short model message.

    Diagnostic-only: mirrors ``HttpOpenCodeClient.send_message`` request
    construction. The neutral service-probe text is never logged.
    """
    body: dict[str, object] = {
        "parts": [
            {
                "type": "text",
                "text": (
                    "Диагностическая проверка соединения с моделью. Ответьте одним словом: Готово."
                ),
            }
        ]
    }
    if model and "/" in model:
        provider, _, name = model.partition("/")
        if provider and name:
            body["model"] = {"providerID": provider, "modelID": name}
    return body


async def run_probe_a(argv: dict[str, Any], sink: EventSink, counters: dict[str, Any]) -> int:
    """Probe A child: health, session create, one message on the AA model."""
    from aa.opencode.runtime import LocalOpenCodeRuntime, OpenCodeConfig

    model = (os.environ.get("OPENCODE_MODEL", "") or "").strip()
    base_url = "http://127.0.0.1:" + str(argv["probe_a_port"])
    config = OpenCodeConfig(
        base_url=base_url,
        command=(os.environ.get("OPENCODE_COMMAND", "") or "opencode"),
        workdir=str(REPO_ROOT),
        model=model,
    )
    runtime = LocalOpenCodeRuntime(config, ready_timeout=45.0)
    anchor = threading.Event()
    watchdog = ChildWatchdog(
        probe=PROBE_DIRECT_MODEL,
        schedule=argv["snapshot_schedule"],
        anchor=anchor,
        anchor_wait_timeout_s=15.0,
        sink=sink,
        checkpoint_dir=Path(argv["out_dir"]),
        opencode_base_url=base_url,
        counters=counters,
    )
    watchdog.start()
    sink.emit("probe-start", probe=PROBE_DIRECT_MODEL, model=str(model)[:96])
    counters["last_real_event"] = "probe-start"
    counters["last_real_monotonic"] = time.monotonic()
    exit_code = EXIT_OK
    session_id = ""
    try:
        # 1) Pre-probe: attach vs spawn evidence (production startup path).
        attached, _ = await asyncio.to_thread(_health_probe, base_url, 1.0)
        try:
            started = time.perf_counter()
            await runtime.start()
            health_ok = bool(runtime.running and runtime.ready)
            await runtime.ensure_ready(5.0)
        except Exception as exc:
            sink.emit(
                "server-health",
                ok=False,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
                attached=attached,
                fail_category=type(exc).__name__[:48],
            )
            sink.emit("probe-end", probe=PROBE_DIRECT_MODEL, status="bootstrap-failed")
            return EXIT_BOOTSTRAP_FAILED
        health_elapsed = round((time.perf_counter() - started) * 1000.0, 1)
        counters["server_ready_ms"] = health_elapsed
        counters["last_real_event"] = "server-health"
        counters["last_real_monotonic"] = time.monotonic()
        sink.emit(
            "server-health",
            ok=health_ok,
            elapsed_ms=health_elapsed,
            attached=attached,
        )
        if not health_ok:
            sink.emit("probe-end", probe=PROBE_DIRECT_MODEL, status="bootstrap-failed")
            return EXIT_BOOTSTRAP_FAILED
        try:
            health_phases = await asyncio.to_thread(
                _health_probe_phased, base_url, "GET", "/global/health"
            )
        except Exception:
            health_phases = {}
        if health_phases:
            sink.emit("server-health-phases", **health_phases)

        # 2) Session creation with phase timing (internal id never logged).
        sink.emit("session-create-start", probe=PROBE_DIRECT_MODEL)
        counters["last_real_event"] = "session-create-start"
        counters["last_real_monotonic"] = time.monotonic()
        ok, _provider_error, timings, extra = await asyncio.to_thread(
            _timed_request,
            base_url,
            "POST",
            "/session",
            {"title": "aa-scenario2-diagnostic"},
            10.0,
        )
        session_id = str(extra.get("_id", "")) if ok else ""
        sink.emit(
            "session-create-end",
            ok=ok,
            connect_ms=timings.get("connect_ms"),
            write_ms=timings.get("write_ms"),
            read_ms=timings.get("read_ms"),
            status=timings.get("status"),
            fail_category=str(timings.get("fail_category", ""))[:48],
        )
        counters["last_real_event"] = "session-create-end"
        counters["last_real_monotonic"] = time.monotonic()
        if not ok or not session_id:
            fail = str(timings.get("fail_category", "") or "session-create-failed")[:48]
            sink.emit("probe-end", probe=PROBE_DIRECT_MODEL, status=fail)
            return EXIT_BOOTSTRAP_FAILED

        # 3) One real model message with per-phase timing; anchor snapshots.
        sink.emit("model-send-start", probe=PROBE_DIRECT_MODEL, model=str(model)[:96])
        counters["last_real_event"] = "model-send-start"
        counters["last_real_monotonic"] = time.monotonic()
        anchor.set()
        started = time.perf_counter()
        ok, provider_error, timings, _extra = await asyncio.to_thread(
            _timed_request,
            base_url,
            "POST",
            f"/session/{session_id}/message",
            _probe_a_message_body(model),
            float(argv["probe_a_message_timeout_s"]),
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        fail_category = str(timings.get("fail_category", "") or "")
        if not fail_category and not ok:
            if provider_error:
                fail_category = "provider-error"
            else:
                fail_category = "model-request-failed"
        counters["last_real_event"] = "model-send-end"
        counters["last_real_monotonic"] = time.monotonic()
        sink.emit(
            "model-send-end",
            ok=ok,
            provider_error=provider_error,
            elapsed_ms=elapsed_ms,
            connect_ms=timings.get("connect_ms"),
            write_ms=timings.get("write_ms"),
            read_ms=timings.get("read_ms"),
            status=timings.get("status"),
            served_model=str(_extra.get("served_model", "") or "")[:96],
            fail_category=fail_category,
        )
        if ok:
            sink.emit("probe-end", probe=PROBE_DIRECT_MODEL, status="completed")
            exit_code = EXIT_OK
        elif fail_category in ("inactivity-timeout", "network-error"):
            sink.emit("probe-end", probe=PROBE_DIRECT_MODEL, status=fail_category)
            exit_code = EXIT_STILL_IN_PROGRESS
        else:
            sink.emit("probe-end", probe=PROBE_DIRECT_MODEL, status=fail_category or "failed")
            exit_code = EXIT_CHILD_UNEXPECTED
        return exit_code
    finally:
        # Bounded best-effort cleanup; the production runtime reaps its child.
        if session_id:
            try:
                await asyncio.to_thread(
                    _timed_request,
                    base_url,
                    "POST",
                    f"/session/{session_id}/abort",
                    None,
                    5.0,
                )
            except Exception:
                pass
            try:
                await asyncio.to_thread(
                    _timed_request, base_url, "DELETE", f"/session/{session_id}", None, 5.0
                )
            except Exception:
                pass
        session_id = ""
        try:
            await runtime.stop()
        except Exception:
            pass
        anchor.set()
        sink.emit("child-exit", probe=PROBE_DIRECT_MODEL, exit_code=exit_code)
        watchdog.stop()


# ---------------------------------------------------------------------------
# Probe B (scenario2): the exact production scenario #2 turn
# ---------------------------------------------------------------------------


class _DiagnosticTelegramApi:
    """Deterministic in-process Telegram network seam (no external network).

    Minimal standalone diagnostic mirror of the qualification lane seam:
    the production ``PollingTelegramTransport`` stays entirely real and
    only this Bot API surface is replaced by an in-process recorder.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sent_texts: list[str] = []
        self.chat_actions = 0
        self.pending_updates: list[dict[str, Any]] = []
        self._counter = 0

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        self.calls.append((method, dict(payload)))
        if method == "getMe":
            return {"id": 700000001, "is_bot": True, "username": "aa_scenario2_diag_bot"}
        if method in {
            "deleteWebhook",
            "setMyCommands",
            "setMyDescription",
            "setMyShortDescription",
        }:
            return True
        if method == "getUpdates":
            if self.pending_updates:
                batch = list(self.pending_updates)
                self.pending_updates.clear()
                return batch
            await asyncio.sleep(0.01)
            return []
        if method == "sendChatAction":
            self.chat_actions += 1
            return True
        if method == "sendMessage":
            self._counter += 1
            value = payload.get("text")
            self.sent_texts.append(value if isinstance(value, str) else "")
            return {"message_id": self._counter}
        raise ValueError("unexpected diagnostic Telegram method")

    async def download_file(self, file_path: str) -> bytes:
        raise ValueError("voice files are not used by the scenario2 diagnostic")

    async def send_voice(self, chat_id: int, ogg_bytes: bytes) -> Any:
        raise ValueError("voice replies are not used by the scenario2 diagnostic")


def _reply_signature(reply: str) -> str:
    """Privacy-safe reply signature (hash only; never the text)."""
    import hashlib

    normalized = " ".join((reply or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


async def run_probe_b(argv: dict[str, Any], sink: EventSink, counters: dict[str, Any]) -> int:
    """Probe B child: exact production scenario #2 (unchanged prompt)."""
    from dataclasses import replace

    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.semantic_work_plan import set_stage_observer
    from aa.telegram.transport import PollingTelegramTransport

    base_url = os.environ.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096")
    out_dir = Path(argv["out_dir"])
    counters["snapshot_index"] = 0
    anchor = threading.Event()
    watchdog = ChildWatchdog(
        probe=PROBE_SCENARIO2,
        schedule=argv["snapshot_schedule"],
        anchor=anchor,
        anchor_wait_timeout_s=30.0,
        sink=sink,
        checkpoint_dir=out_dir,
        opencode_base_url=base_url,
        counters=counters,
    )
    watchdog.start()

    stage_state = {"prev": "", "start": 0.0}

    def observer(stage: str) -> None:
        now = time.monotonic()
        previous = stage_state["prev"]
        if previous and stage_state["start"]:
            try:
                sink.emit(
                    "stage-end",
                    stage=str(previous)[:64],
                    elapsed_ms=round(max(0.0, now - stage_state["start"]) * 1000.0, 1),
                )
            except Exception:
                pass
        stage_state["prev"] = stage
        stage_state["start"] = now
        try:
            sink.emit("stage-start", stage=str(stage)[:64])
        except Exception:
            pass
        counters["last_stage"] = str(stage)[:64]
        counters["last_stage_monotonic"] = now
        counters["last_stage_at_s"] = round(now - (counters.get("probe_start") or now), 2)
        counters["last_real_event"] = "stage-start:" + str(stage)[:48]
        counters["last_real_monotonic"] = now

    settings = replace(
        Settings.from_env(None),
        bot_session_duration_seconds=0.0,
        typing_heartbeat_seconds=0.05,
    )
    api = _DiagnosticTelegramApi()
    transport = PollingTelegramTransport(
        token="scenario2-diagnostic-in-process-seam",
        api=api,
        poll_timeout_seconds=0,
        retry_base_delay_seconds=0.01,
        retry_max_delay_seconds=0.05,
    )
    app = Application(settings, transport=transport)
    counters["probe_start"] = time.monotonic()
    sink.emit("probe-start", probe=PROBE_SCENARIO2, family=SCENARIO_2_FAMILY)
    counters["last_real_event"] = "probe-start"
    counters["last_real_monotonic"] = time.monotonic()
    exit_code = EXIT_OK
    try:
        sink.emit("bootstrap-start", probe=PROBE_SCENARIO2)
        counters["last_real_event"] = "bootstrap-start"
        counters["last_real_monotonic"] = time.monotonic()
        started = time.perf_counter()
        try:
            await app.start()
        except Exception as exc:
            sink.emit(
                "bootstrap-end",
                ok=False,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
                fail_category=type(exc).__name__[:48],
            )
            sink.emit("probe-end", probe=PROBE_SCENARIO2, status="bootstrap-failed")
            return EXIT_BOOTSTRAP_FAILED
        counters["last_real_event"] = "bootstrap-end"
        counters["last_real_monotonic"] = time.monotonic()
        graph = app.graph_runtime
        sink.emit(
            "bootstrap-end",
            ok=True,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
            opencode_ready=bool(app.opencode_runtime.running and app.opencode_runtime.ready),
            graph_running=bool(graph is not None and graph.running),
        )
        watchdog.audit_provider = lambda: list(app.opencode_runtime.client.request_latency_audit)
        set_stage_observer(observer)
        before = len(api.sent_texts)
        sink.emit(
            "scenario-start",
            family=SCENARIO_2_FAMILY,
            scenario_index=2,
            chat_id=SCENARIO_2_CHAT_ID,
        )
        counters["last_real_event"] = "scenario-start"
        counters["last_real_monotonic"] = time.monotonic()
        raw = {
            "update_id": 930103,
            "message": {
                "message_id": 2,
                "date": 1,
                "chat": {"id": SCENARIO_2_CHAT_ID, "type": "private"},
                "text": SCENARIO_2_PROMPT,
            },
        }
        api.pending_updates.append(raw)
        scenario_started = time.monotonic()
        anchor.set()
        cutoff = float(argv["cutoff_s"])
        while len(api.sent_texts) <= before:
            counters["typing_sends"] = api.chat_actions
            elapsed = time.monotonic() - scenario_started
            if elapsed >= cutoff:
                sink.emit(
                    "delivery-timeout",
                    cutoff_s=cutoff,
                    last_stage=str(counters.get("last_stage", "") or "")[:64],
                    seconds_since_last_stage=max(
                        0.0, time.monotonic() - float(counters.get("last_stage_monotonic") or 0)
                    ),
                    elapsed_ms=round((time.monotonic() - scenario_started) * 1000.0, 1),
                )
                sink.emit(
                    "probe-end", probe=PROBE_SCENARIO2, status="still-in-progress-at-probe-end"
                )
                exit_code = EXIT_STILL_IN_PROGRESS
                return exit_code
            await asyncio.sleep(0.02)
        elapsed_ms = round((time.monotonic() - scenario_started) * 1000.0, 1)
        counters["delivered"] = True
        counters["last_real_event"] = "delivery"
        counters["last_real_monotonic"] = time.monotonic()
        reply = api.sent_texts[-1]
        sink.emit(
            "delivery",
            elapsed_ms=elapsed_ms,
            reply_len=len(reply),
            reply_signature=_reply_signature(reply),
            typing_sends=api.chat_actions,
        )
        stage_values = _collect_stage_telemetry(graph)
        sink.emit("scenario-end", family=SCENARIO_2_FAMILY, stage_values=len(stage_values))
        sink.emit("probe-end", probe=PROBE_SCENARIO2, status="delivered")
        exit_code = EXIT_OK
        return exit_code
    finally:
        set_stage_observer(None)
        try:
            await app.stop()
        except Exception:
            pass
        anchor.set()
        sink.emit("child-exit", probe=PROBE_SCENARIO2, exit_code=exit_code)
        watchdog.stop()


def _collect_stage_telemetry(graph: Any) -> list[dict[str, Any]]:
    """Sanitized per-stage turn telemetry (values only, no stage text)."""
    values: list[dict[str, Any]] = []
    if graph is None:
        return values
    try:
        telemetry = graph.last_telemetry_for_thread(graph.thread_id(SCENARIO_2_CHAT_ID))
        if isinstance(telemetry, dict):
            for name, value in telemetry.items():
                if isinstance(value, (int, float, bool)):
                    values.append({"stage": str(name)[:48], "value": value})
    except Exception:
        return values
    return values


def _child_main(argv: dict[str, Any]) -> int:
    """Child entry: one probe under its independent watchdog thread."""
    out_dir = Path(argv["out_dir"])
    probe = str(argv["probe"])
    sink = EventSink(out_dir / f"probe-{probe}-events.jsonl", process_start=time.monotonic())
    _install_stack_handler(out_dir / f"stacks-{probe}.txt")
    counters: dict[str, Any] = {"snapshot_index": 0}
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if probe == PROBE_DIRECT_MODEL:
            return int(loop.run_until_complete(run_probe_a(argv, sink, counters)))
        if probe == PROBE_SCENARIO2:
            return int(loop.run_until_complete(run_probe_b(argv, sink, counters)))
        return EXIT_REPORT_FAILED
    except Exception as exc:
        try:
            frames = [sanitize_frame(fr.f_code) for fr in traceback.extract_tb(sys.exc_info()[2])][
                -4:
            ]
        except Exception:
            frames = []
        sink.emit(
            "child-unexpected",
            category=type(exc).__name__[:48],
            frames=frames,
        )
        return EXIT_CHILD_UNEXPECTED
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Issue progress publisher (GitHub issues; ONE idempotent comment per run)
# ---------------------------------------------------------------------------


def comment_marker(run_id: str) -> str:
    """Idempotence marker for exactly one diagnostic comment per run."""
    return f"<!-- aa-scenario2-diagnostic run={run_id} -->"


def _body_line_has_text(line: str) -> bool:
    """Fail-closed scan of one rendered ``- key: value`` body line."""
    token = line.lstrip(" -*").split(":", 1)[0].strip().casefold()
    return token in _FORBIDDEN_KEYS


class IssueProgressPublisher:
    """Posts or updates ONE sanitized diagnostic comment (fail-soft)."""

    MIN_INTERVAL_S = 20.0
    _API_ROOT = "https://api.github.com"

    def __init__(self, owner: str, repo: str, issue: int, token: str, run_id: str = ""):
        self.owner = owner or DEFAULT_OWNER
        self.repo = repo or DEFAULT_REPO
        self.issue = int(issue or DEFAULT_ISSUE)
        self.token = (token or "").strip()
        self.run_id = run_id or os.environ.get("GITHUB_RUN_ID", "") or "local"
        self._comment_id: int | None = None
        self._last_post = 0.0

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[bool, Any]:
        if not self.token:
            return False, None
        url = f"{self._API_ROOT}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=15.0) as response:
                body = response.read()
        except (OSError, ValueError):
            return False, None
        if not body:
            return True, None
        try:
            return True, json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return True, None

    def ensure_comment(self) -> int | None:
        """Find this run's existing diagnostic comment (bounded, idempotent)."""
        if self._comment_id is not None:
            return self._comment_id
        marker = comment_marker(self.run_id)
        ok, comments = self._request(
            "GET",
            f"/repos/{self.owner}/{self.repo}/issues/{self.issue}/comments?per_page=100",
        )
        if not ok or not isinstance(comments, list):
            return None
        for comment in comments[:100]:
            if not isinstance(comment, dict):
                continue
            body = comment.get("body")
            if isinstance(body, str) and marker in body:
                comment_id = comment.get("id")
                if isinstance(comment_id, int):
                    self._comment_id = comment_id
                    return comment_id
        return None

    def publish(self, title: str, body: str) -> bool:
        """Create once, then update the same comment (throttled, fail-soft)."""
        safe_title = title.replace("|", "/").replace("\n", " ")[:64]
        stamped = f"{comment_marker(self.run_id)}\n**{safe_title}**\n\n{body}\n"
        assert_no_text_leak({"comment_title": safe_title})
        try:
            assert_no_text_leak({"comment_body": body})
            if any(_body_line_has_text(line) for line in body.splitlines()):
                raise ValueError("comment body carries a text-keyed line")
        except ValueError:
            body = "- text-free renditions only:\n- [text suppressed by sanitizer]\n"
            stamped = f"{comment_marker(self.run_id)}\n**{safe_title}**\n\n{body}\n"
        now = time.monotonic()
        if now - self._last_post < self.MIN_INTERVAL_S:
            return True
        self._last_post = now
        comment_id = self.ensure_comment()
        if comment_id is not None:
            ok, _ = self._request(
                "PATCH",
                f"/repos/{self.owner}/{self.repo}/issues/{self.issue}/comments/{comment_id}",
                {"body": stamped},
            )
            if not ok:
                self._comment_id = None
            return ok
        ok, created = self._request(
            "POST",
            f"/repos/{self.owner}/{self.repo}/issues/{self.issue}/comments",
            {"body": stamped},
        )
        if ok and isinstance(created, dict) and isinstance(created.get("id"), int):
            self._comment_id = int(created["id"])
        return ok


# ---------------------------------------------------------------------------
# Parent mode: child supervision, cadence comments and the final report
# ---------------------------------------------------------------------------


@dataclass
class ProbeEvidence:
    """Privacy-safe evidence state assembled by the supervisor thread."""

    probe: str
    status: str = "not-run"
    exit_code: int | None = None
    killed: bool = False
    last_stage: str = ""
    last_stage_at_s: float | None = None
    seconds_since_last_stage: float | None = None
    last_real_event: str = ""
    last_real_ts: str = ""
    delivered: bool = False
    delivered_elapsed_ms: float | None = None
    reply_len: int | None = None
    reply_signature: str = ""
    fail_category: str = ""
    server_ready_ms: float | None = None
    connect_ms: float | None = None
    write_ms: float | None = None
    read_ms: float | None = None
    model_elapsed_ms: float | None = None
    timeline: list[dict[str, Any]] = field(default_factory=list)
    snapshots: int = 0
    raw_lines: int = 0


@dataclass
class ProbePlan:
    """One probe's bounded supervision parameters."""

    probe: str
    schedule: tuple[float, ...]
    child_cutoff_s: float


def _spawn_child(*, plan: ProbePlan, out_dir: Path) -> subprocess.Popen:
    """Start one probe child in its own process group (independent)."""
    child_argv = json.dumps(
        {
            "probe": plan.probe,
            "out": str(out_dir),
            "snapshots": list(plan.schedule),
            "cutoff": plan.child_cutoff_s,
        }
    )
    env = os.environ.copy()
    env["SCENARIO2_DIAG_ARGS"] = child_argv
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(  # noqa: S603 (fixed argv, no shell)
        [sys.executable, str(Path(__file__).resolve()), "--child"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def child_argv_from_env() -> dict[str, Any]:
    """Decode the parent-provided child argv (single JSON env value)."""
    try:
        decoded = json.loads(os.environ.get("SCENARIO2_DIAG_ARGS", "") or "{}")
    except ValueError:
        decoded = {}
    return {
        "probe": str(decoded.get("probe", "")),
        "out_dir": str(decoded.get("out", "")),
        "snapshot_schedule": tuple(float(v) for v in decoded.get("snapshots", ())),
        "cutoff_s": float(decoded.get("cutoff", 0.0)),
        "probe_a_port": str(os.environ.get("SCENARIO2_DIAG_PROBE_A_PORT", DEFAULT_PROBE_A_PORT)),
        "probe_a_message_timeout_s": float(
            os.environ.get("SCENARIO2_DIAG_PROBE_A_MESSAGE_TIMEOUT_S", "")
            or DEFAULT_PROBE_A_MESSAGE_TIMEOUT_S
        ),
    }


def _reader_loop(stdout: Any, evidence: ProbeEvidence, sink: EventSink) -> None:
    """Consume child stdout JSONL into the Actions log + probe state."""
    try:
        for line in stdout:
            line = line.strip()
            if not line:
                continue
            sink.passthrough(line)
            try:
                payload = json.loads(line)
            except ValueError:
                evidence.raw_lines += 1
                continue
            kind = str(payload.get("event", ""))
            if kind in REAL_ACTIVITY_KINDS:
                evidence.last_real_event = kind[:48]
                evidence.last_real_ts = str(payload.get("ts", ""))
            if kind == "snapshot":
                evidence.snapshots += 1
                stage = str(payload.get("stage", "") or "")[:64]
                if stage:
                    evidence.last_stage = stage
                evidence.last_stage_at_s = payload.get("stage_at_s")
                evidence.seconds_since_last_stage = payload.get("seconds_since_last_stage")
                if len(evidence.timeline) < 48:
                    evidence.timeline.append(
                        {
                            "at_s": payload.get("at_s"),
                            "stage": evidence.last_stage,
                            "since_stage_s": evidence.seconds_since_last_stage,
                            "typing_sends": payload.get("typing_sends", 0),
                            "delivered": bool(payload.get("delivered", False)),
                        }
                    )
            elif kind == "server-health":
                evidence.server_ready_ms = payload.get("elapsed_ms")
                if payload.get("ok") is False:
                    evidence.fail_category = "server-health-failed"
            elif kind == "model-send-end":
                evidence.model_elapsed_ms = payload.get("elapsed_ms")
                evidence.connect_ms = payload.get("connect_ms")
                evidence.write_ms = payload.get("write_ms")
                evidence.read_ms = payload.get("read_ms")
                if not payload.get("ok"):
                    evidence.fail_category = str(
                        payload.get("fail_category", "") or "model-request-failed"
                    )[:48]
            elif kind == "delivery":
                evidence.delivered = True
                evidence.delivered_elapsed_ms = payload.get("elapsed_ms")
                evidence.reply_len = payload.get("reply_len")
                evidence.reply_signature = str(payload.get("reply_signature", ""))[:32]
            elif kind == "bootstrap-end" and payload.get("ok") is False:
                evidence.fail_category = (
                    "bootstrap:" + str(payload.get("fail_category", "unknown"))[:48]
                )
                evidence.status = "bootstrap-failed"
            elif kind == "delivery-timeout":
                evidence.fail_category = "still-in-progress-at-probe-end"
            elif kind == "probe-end":
                evidence.status = str(payload.get("status", ""))[:48]
            elif kind == "child-exit":
                break
            if evidence.raw_lines > 200:  # bounded raw non-JSON capture
                break
    finally:
        try:
            stdout.close()
        except Exception:
            pass


def _kill_group(proc: subprocess.Popen, grace_s: float) -> bool:
    """SIGTERM then SIGKILL the child's whole process group."""
    killed = False
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
            killed = True
        except OSError:
            pass
    try:
        proc.wait(timeout=max(1.0, grace_s))
        return killed
    except TimeoutError:
        pass
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            killed = True
        except OSError:
            pass
    try:
        proc.wait(timeout=max(1.0, grace_s))
    except TimeoutError:
        killed = True
    return killed


def _send_sigusr1(proc: subprocess.Popen) -> None:
    """Request a whole-process faulthandler dump inside the child."""
    try:
        proc.send_signal(signal.SIGUSR1)
    except (ValueError, OSError):
        pass


def _final_supervisor_snapshot(
    out_dir: Path, proc: subprocess.Popen, evidence: ProbeEvidence, *, kind: str
) -> dict[str, Any]:
    """Terminal snapshot collected by the supervisor (independent of child)."""
    _send_sigusr1(proc)
    time.sleep(1.5)
    payload = {
        "type": kind,
        "ts": now_iso(),
        "probe": evidence.probe,
        "child_status": ("running" if proc.poll() is None else "exited"),
        "opencode_processes": count_opencode_processes(),
        "sockets": summarize_loopback_sockets(),
    }
    try:
        (out_dir / f"{kind}.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass
    return payload


def _probe_comment_lines(evidence: ProbeEvidence) -> list[str]:
    """Compact sanitized markdown for one probe update."""
    rows: list[str] = [f"- component: `{evidence.probe}`"]
    rows.append(f"- status: `{evidence.status or 'unknown'}`")
    if evidence.server_ready_ms is not None:
        rows.append(f"- server_ready_ms: {evidence.server_ready_ms}")
    if evidence.model_elapsed_ms is not None:
        rows.append(f"- model_request_total_ms: {evidence.model_elapsed_ms}")
        for phase in ("connect_ms", "write_ms", "read_ms"):
            value = getattr(evidence, phase)
            if value is not None:
                rows.append(f"- {phase}: {value}")
    if evidence.delivered_elapsed_ms is not None:
        rows.append(f"- delivery_ms: {evidence.delivered_elapsed_ms}")
        if evidence.reply_len is not None:
            rows.append(f"- reply_len: {evidence.reply_len}")
    if evidence.last_stage:
        if evidence.seconds_since_last_stage is not None:
            rows.append(
                f"- last_stage: `{evidence.last_stage}` "
                f"(stage age at snapshot: {evidence.seconds_since_last_stage}s)"
            )
        else:
            rows.append(f"- last_stage: `{evidence.last_stage}`")
    if evidence.fail_category:
        rows.append(f"- fail_category: `{evidence.fail_category}`")
    if evidence.last_real_event:
        rows.append(f"- last_real_event: `{evidence.last_real_event}`")
    rows.append(f"- snapshots: {evidence.snapshots}")
    rows.append(f"- killed: {str(evidence.killed).lower()}")
    return rows


def _publish_probe_update(publisher: IssueProgressPublisher, evidence: ProbeEvidence) -> None:
    body = "\n".join(_probe_comment_lines(evidence))
    publisher.publish(f"Scenario #2 diagnostic: {evidence.probe}", body)


def _run_one_probe(
    plan: ProbePlan,
    *,
    out_dir: Path,
    sink: EventSink,
    publisher: IssueProgressPublisher | None,
) -> ProbeEvidence:
    """Supervise one probe child end-to-end (bounded, isolated)."""
    evidence = ProbeEvidence(probe=plan.probe)
    evidence.status = "running"
    sink.emit(
        "supervisor-probe-start",
        probe=plan.probe,
        child_cutoff_s=plan.child_cutoff_s,
        snapshot_schedule=list(plan.schedule),
    )
    proc = _spawn_child(plan=plan, out_dir=out_dir)
    reader = threading.Thread(target=_reader_loop, args=(proc.stdout, evidence, sink), daemon=True)
    reader.start()
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    sink.emit("supervisor-child-spawn", probe=plan.probe, has_group=pgid is not None)
    started = time.monotonic()
    last_stack_request = 0.0
    stack_dumps = 0
    last_publish = started
    supervision_limit = plan.child_cutoff_s + CHILD_CUTOFF_SLACK_S
    while proc.poll() is None:
        now = time.monotonic()
        if now - started >= supervision_limit:
            sink.emit(
                "supervisor-cutoff",
                probe=plan.probe,
                elapsed_s=round(now - started, 1),
                limit_s=round(supervision_limit, 1),
            )
            _final_supervisor_snapshot(
                out_dir, proc, evidence, kind=f"supervisor-final-snapshot-{plan.probe}"
            )
            evidence.killed = _kill_group(proc, DEFAULT_TERMINATE_GRACE_S)
            if not evidence.delivered and plan.probe == PROBE_SCENARIO2:
                evidence.status = "still-in-progress-at-probe-end"
            elif evidence.status in ("", "running"):
                evidence.status = f"child-exit-{proc.returncode}"
            break
        if stack_dumps < MAX_STACK_DUMPS and now - last_stack_request >= STACK_DUMP_EVERY_S:
            _send_sigusr1(proc)
            stack_dumps += 1
            last_stack_request = now
        if publisher is not None and now - last_publish >= COMMENT_CADENCE_S:
            _publish_probe_update(publisher, evidence)
            last_publish = now
        time.sleep(SUPERVISOR_POLL_S)
    if proc.poll() is None:
        _kill_group(proc, DEFAULT_TERMINATE_GRACE_S)
    reader.join(timeout=3.0)
    evidence.exit_code = proc.returncode
    if evidence.status in ("", "running"):
        evidence.status = f"child-exit-{proc.returncode}"
    if publisher is not None:
        _publish_probe_update(publisher, evidence)
    sink.emit(
        "supervisor-probe-end",
        probe=plan.probe,
        status=evidence.status,
        exit_code=None if evidence.exit_code is None else int(evidence.exit_code),
        killed=evidence.killed,
        snapshots=evidence.snapshots,
    )
    return evidence


# ---------------------------------------------------------------------------
# Classification and the compact root-cause report
# ---------------------------------------------------------------------------


def classify_cause(probe_a: ProbeEvidence, probe_b: ProbeEvidence) -> str:
    """Evidence-driven classification; 'stage-unobserved' is named, never guessed.

    Direct-model evidence comes first: if a direct model call itself stalls
    or errors, the provider/OpenCode boundary is proven without blaming
    retrieval downstream.
    """
    if probe_a.status in ("bootstrap-failed", "not-run"):
        return "local-opencode-server"
    if probe_a.status in ("inactivity-timeout", "network-error", "provider-error"):
        return "model-provider"
    if probe_b.status == "bootstrap-failed":
        return "bootstrap-dependency"
    if probe_b.delivered:
        return "delivered"
    if probe_b.fail_category.startswith("bootstrap:"):
        return "bootstrap-dependency"
    stage = (probe_b.last_stage or "").strip()
    if not stage:
        return "stage-unobserved"
    return _STAGE_COMPONENT.get(stage.casefold(), "stage-unobserved")


_NEXT_FIX: dict[str, str] = {
    "local-opencode-server": (
        "Inspect the supervised `opencode serve` spawn/health path "
        "(LocalOpenCodeRuntime) on this runner: verify the pinned binary "
        "version and full health timing; only then consider a bounded "
        "attach fallback. No product timeout changes."
    ),
    "model-provider": (
        "Direct model call itself stalled/failed, so investigate the "
        "provider/model boundary for the approved AA bot model "
        "(quota/429 handling and served-model identity) before touching "
        "retrieval. No product timeout changes."
    ),
    "bootstrap-dependency": (
        "Fix the failing bootstrap step first (corpus restore or index "
        "build per the workflow step name in the run log); do not "
        "misattribute this as case-2 execution."
    ),
    "aa-graph-planner": (
        "Add bounded planner structured-output attempt telemetry and "
        "inspect the in-flight planner call; the stall was observed before "
        "downstream stages could start."
    ),
    "aa-retrieval-index": (
        "Profile the RAM E5/FAISS index open; evidence shows the index "
        "stage as the last observed activity."
    ),
    "aa-retrieval": (
        "Profile the in-flight retrieval call and RRF/selection path; "
        "evidence shows delivery stalled inside or after retrieval."
    ),
    "aa-graph-answer": (
        "Inspect the in-flight answer synthesis call and its structured output handling."
    ),
    "aa-graph-verifier": (
        "Inspect the in-flight verifier turn and its structured output handling."
    ),
    "aa-graph-judge": ("Inspect the in-flight whole-turn judge call."),
    "aa-graph-finalization": (
        "Inspect answer finalization/envelope handling at the observed stage."
    ),
    "aa-graph-orchestration": (
        "Inspect the LangGraph turn-boundary transitions captured in the "
        "snapshot stacks before the first substantive stage."
    ),
    "telegram-dispatch": (
        "Inspect the in-process sendMessage serialization/wait path in the "
        "production PollingTelegramTransport."
    ),
    "stage-unobserved": (
        "The turn produced NO observed stage marker. Use the captured "
        "thread stacks/task states and OpenCode request audit to identify "
        "the exact in-flight call; rerun with deeper stage hooks only if "
        "stacks are not enough. Do not assign an upstream cause."
    ),
    "still-in-progress-at-probe-end": (
        "Extend this diagnostic observation window with one rerun; the "
        "turn may still genuinely progress after the cutoff. INCONCLUSIVE "
        "is not a product FAIL."
    ),
    "delivered": (
        "No isolated fix is indicated: the scenario delivered within the "
        "diagnostic window; attach this timing evidence to the broad "
        "repair analysis (kodmial/aa#417)."
    ),
}


def _run_url() -> str:
    """Actions run URL (no credentials in URL)."""
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY", "kodmial/aa")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    return f"{server}/{repository}/actions/runs/{run_id}".strip()


def build_final_report(meta: dict[str, Any], probe_a: ProbeEvidence, probe_b: ProbeEvidence) -> str:
    """Compact actionable root-cause report (privacy-safe, one screen)."""
    classification = classify_cause(probe_a, probe_b)
    next_fix = _NEXT_FIX.get(
        classification, "One more focused diagnostic run with deeper evidence is required."
    )
    lines = [
        "Single scenario #2 (`substantive-drinking`) diagnostic; NOT a Gate C "
        "PASS/FAIL, no full 18-case rerun and no product criterion changed.",
        "",
        "The bounded diagnostic observation window is NOT a product "
        "qualification timeout: `still-in-progress-at-probe-end` means "
        "INCONCLUSIVE, never a product FAIL.",
        f"Run: `{meta['run_url']}` (exact SHA `{meta['main_sha']}`)",
        "",
        "### Probe A — direct model (OpenCode server + approved model)",
        *_probe_comment_lines(probe_a),
        "",
        "### Probe B — exact production scenario #2",
        *_probe_comment_lines(probe_b),
        "",
        "### Root-cause classification",
        f"- classification: `{classification}`",
    ]
    if probe_b.last_stage:
        lines.append(f"- last observed stage: `{probe_b.last_stage}`")
    if classification == "stage-unobserved":
        lines.append("- stage=unobserved (explicit: no stage markers arrived)")
    if probe_b.last_real_event:
        lines.append(f"- last confirmed REAL activity: `{probe_b.last_real_event}`")
    lines.append(f"- snapshots captured: A={probe_a.snapshots}, B={probe_b.snapshots}")
    lines.append(
        f"- child killed by supervisor: A={str(probe_a.killed).lower()}, "
        f"B={str(probe_b.killed).lower()}"
    )
    lines.append("")
    lines.append("### Narrow next step")
    lines.append(next_fix)
    return "\n".join(lines)


def _evidence_dict(evidence: ProbeEvidence) -> dict[str, Any]:
    """Privacy-safe evidence serialization."""
    return {
        "probe": evidence.probe,
        "status": evidence.status,
        "exit_code": evidence.exit_code,
        "killed": evidence.killed,
        "last_stage": evidence.last_stage,
        "last_real_event": evidence.last_real_event,
        "delivered": evidence.delivered,
        "delivered_elapsed_ms": evidence.delivered_elapsed_ms,
        "fail_category": evidence.fail_category,
        "server_ready_ms": evidence.server_ready_ms,
        "model_request_total_ms": evidence.model_elapsed_ms,
        "snapshots": evidence.snapshots,
        "timeline": list(evidence.timeline)[:24],
    }


def git_head_sha() -> str:
    """Exact checked-out SHA when git metadata is available."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown"


def parent_main(args: argparse.Namespace) -> int:
    """Supervise both probes, collect evidence and publish the report."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": SCHEMA_VERSION,
        "main_sha": args.main_sha or git_head_sha(),
        "run_id": str(args.run_id),
        "target_issue": int(args.issue),
        "target_owner": args.owner,
        "target_repo": args.repo,
        "run_url": _run_url(),
        "diagnostic_cutoff_s": float(args.diagnostic_cutoff_s),
        "snapshots": list(parse_schedule(args.snapshots)),
    }
    (out_dir / "run-meta.json").write_text(
        json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    sink = EventSink(out_dir / "supervisor-events.jsonl", echo=False)
    sink.emit("supervisor-start", run_id=meta["run_id"], sha=meta["main_sha"])

    token = (os.environ.get("GITHUB_TOKEN", "") or "").strip()
    publisher = IssueProgressPublisher(
        args.owner, args.repo, args.issue, token, run_id=str(args.run_id)
    )
    if token:
        publisher.publish(
            "Scenario #2 diagnostic started",
            f"- run: `{meta['run_url']}`\n- sha: `{meta['main_sha']}`",
        )

    probe_a = ProbeEvidence(probe=PROBE_DIRECT_MODEL, status="not-run")
    probe_b = ProbeEvidence(probe=PROBE_SCENARIO2, status="not-run")
    os.environ["SCENARIO2_DIAG_PROBE_A_PORT"] = str(args.probe_a_port)
    os.environ["SCENARIO2_DIAG_PROBE_A_MESSAGE_TIMEOUT_S"] = str(args.probe_a_message_timeout_s)
    try:
        if args.probe in ("both", PROBE_DIRECT_MODEL):
            probe_a = _run_one_probe(
                ProbePlan(
                    probe=PROBE_DIRECT_MODEL,
                    schedule=parse_schedule(args.probe_a_snapshots),
                    child_cutoff_s=float(args.probe_a_cutoff_s),
                ),
                out_dir=out_dir,
                sink=sink,
                publisher=publisher,
            )
        if args.probe in ("both", PROBE_SCENARIO2):
            probe_b = _run_one_probe(
                ProbePlan(
                    probe=PROBE_SCENARIO2,
                    schedule=parse_schedule(args.snapshots),
                    child_cutoff_s=float(args.diagnostic_cutoff_s),
                ),
                out_dir=out_dir,
                sink=sink,
                publisher=publisher,
            )
    finally:
        sink.emit("supervisor-stray-check", opencode_processes_leftover=count_opencode_processes())

    report = build_final_report(meta, probe_a, probe_b)
    (out_dir / "final-report.md").write_text(report + "\n", encoding="utf-8")
    result = {
        "schema": SCHEMA_VERSION,
        "run_id": meta["run_id"],
        "main_sha": meta["main_sha"],
        "run_url": meta["run_url"],
        "target_issue": meta["target_issue"],
        "target_owner": meta["target_owner"],
        "target_repo": meta["target_repo"],
        "probe_direct_model": _evidence_dict(probe_a),
        "probe_scenario2": _evidence_dict(probe_b),
        "classification": classify_cause(probe_a, probe_b),
        "next_fix": _NEXT_FIX.get(classify_cause(probe_a, probe_b), ""),
        "diagnostic_note": (
            "Diagnostic observation window only; not a product qualification "
            "timeout and not a Gate C PASS/FAIL. Still-in-progress at the end "
            "is INCONCLUSIVE, never a product FAIL."
        ),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if token:
        publisher.publish("Scenario #2 diagnostic FINAL REPORT", report)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if step_summary:
        try:
            with Path(step_summary).open("a", encoding="utf-8") as stream:
                stream.write("## AA Gate C scenario2 diagnostic\n\n```markdown\n")
                stream.write(report)
                stream.write("\n```\n")
        except OSError:
            pass
    sink.emit("supervisor-report-published", target_issue=meta["target_issue"])
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "final_report": str(out_dir / "final-report.md"),
                "artifact": f"aa-gate-c-scenario2-diagnostic-{meta['run_id']}-scenario2",
            }
        ),
        flush=True,
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """CLI surface for the parent mode (child params travel via env)."""
    parser = argparse.ArgumentParser(
        description="Isolated Gate C scenario #2 diagnostic (issue #418)."
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--probe",
        choices=("both", PROBE_DIRECT_MODEL, PROBE_SCENARIO2),
        default="both",
    )
    parser.add_argument("--main-sha", default="", help="Exact SHA under test")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument("--issue", type=int, default=DEFAULT_ISSUE)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out-dir", default="diagnostic-scenario2-out")
    parser.add_argument("--snapshots", default="10,25,45,75,110")
    parser.add_argument("--diagnostic-cutoff-s", type=float, default=DEFAULT_DIAGNOSTIC_CUTOFF_S)
    parser.add_argument("--probe-a-snapshots", default="5,15,30,50,70")
    parser.add_argument("--probe-a-cutoff-s", type=float, default=DEFAULT_PROBE_A_CUTOFF_S)
    parser.add_argument("--probe-a-port", default=DEFAULT_PROBE_A_PORT)
    parser.add_argument(
        "--probe-a-message-timeout-s", type=float, default=DEFAULT_PROBE_A_MESSAGE_TIMEOUT_S
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: dispatch to the child or the parent mode."""
    args = build_parser().parse_args(argv)
    if args.child:
        child = child_argv_from_env()
        if child["probe"] not in (PROBE_DIRECT_MODEL, PROBE_SCENARIO2):
            return EXIT_REPORT_FAILED
        return _child_main(child)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
