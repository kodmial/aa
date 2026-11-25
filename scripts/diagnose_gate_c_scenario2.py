#!/usr/bin/env python3
"""Gate C scenario #2 focused diagnostic (issues #418, #424) — P0 isolated probe.

DIAGNOSTIC ONLY. This script never decides Gate A-F qualification, never
touches the product criteria of issue #7 and never reruns the full 18-case
Gate C. It runs exactly two separable probes against the real production
runtime on the exact checked-out SHA:

- Probe A (``direct-model``): real local ``opencode serve`` (spawned by the
  production ``LocalOpenCodeRuntime``), the approved AA model pointer, one
  simple session + one short message. Per-phase HTTP timing (connect /
  write / read) is captured so a direct model/provider error is PROVEN
  from a real response, never guessed from a wall clock;
- Probe B (``scenario2``): the exact production scenario #2
  (``substantive-drinking``) through the unchanged production boundary
  (``PollingTelegramTransport`` -> ``Application`` -> per-chat dispatcher ->
  LangGraph planner/retrieval/answer/verifier), using the deterministic
  in-process Telegram ``getUpdates``/``sendMessage`` seam (NO external
  Telegram network) and the unchanged scenario #2 prompt.

kodmial/aa#424 policy (hard contract):

- There is **NO per-answer wall-clock cutoff anywhere**: no 45s message
  timeout, no 75s direct-probe window and no 130s scenario window. A real
  model call that continues past 45/75/130/300/900 s is OBSERVED, not
  killed. A long pending OpenCode response with no events is an
  UNOBSERVED in-flight request, not proof of provider failure.
- Observation is progress/liveness-driven. It ends only on: delivery, a
  PROVEN provider error (real error payload from a real response), a
  PROVEN local server failure (spawn/health/transport evidence), a
  diagnostic-harness fault, or the single configurable campaign physical
  allowance. A typing/polling heartbeat alone is never treated as
  progress.
- The one finite operational boundary is the **campaign allowance** (one
  job-level ceiling plus a finalisation reserve), default 1800 s — the
  same order as the AA qualification's real-progress scenario inactivity
  allowance (``live_watchdog.DEFAULT_SCENARIO_INACTIVITY_S``) — reconciled
  with the remaining Actions job budget so artifact upload always
  completes. A turn at the allowance is reported
  ``unresolved-at-campaign-end``: evidence plus an optional resumable
  experiment, never a ``model-provider`` root cause.
- Bounded diagnostics-only infrastructure bounds remain explicitly typed:
  OpenCode spawn/bootstrap/connect health and observation probes are
  finite. Answer/model calls have NO total-response deadline; they mirror
  production ``HttpOpenCodeClient`` with a separate per-socket inactivity
  bound so a silent provider connection cannot block forever.
- Truthful verdicts only: ``delivered``, ``provider-error-proven``,
  ``server-error-proven``, ``harness-error``, ``unresolved-at-campaign-end``,
  ``boundary-localized-root-unknown`` (BOUNDARY LOCALIZED / ROOT UNKNOWN),
  ``stage-unobserved``. An in-flight ``opencode`` stage is NEVER mapped
  to a proven provider root cause.

Privacy: no user text, no book text, no model payloads, no tokens, no
session ids, no URLs with credentials and no raw environment is ever
printed or written. Events carry ids, categories, counts, latencies and
code identifiers only. The scenario prompt itself is never logged.

Progressive evidence: line-buffered timestamped JSONL lifecycle events
stream to stdout (the GitHub Actions log) and to privacy-safe artifact
files. The phase anchor turns into a periodic cadence: an in-child
watchdog thread snapshots full in-process state (threads, tasks, last
real progress vs typing heartbeat, OpenCode request audit, socket table)
plus sanitized incremental OpenCode internal status (sessions/messages/
part-kind counts when the pinned server version supports them) on
cadence; independent whole-process faulthandler stacks are requested by
the supervisor via SIGUSR1 after the child published its readiness
sentinel.

Usage (workflow):
  python scripts/diagnose_gate_c_scenario2.py \
    --main-sha <sha> --run-id <run_id> --issue 418 \
    --campaign-allowance-s 1800 --job-timeout-minutes 60 \
    --job-started-epoch <job_start_epoch> \
    --out-dir diagnostic-scenario2-out
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import faulthandler
import hashlib
import http.client
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
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

SCHEMA_VERSION = "aa-gate-c-scenario2-diagnostic/2"
PROBE_DIRECT_MODEL = "direct-model"
PROBE_SCENARIO2 = "scenario2"
PROBE_SCENARIO2_OFF = "scenario2-off"
PROBE_SCENARIO2_ON = "scenario2-on"
SCENARIO2_PROBES = frozenset({PROBE_SCENARIO2, PROBE_SCENARIO2_OFF, PROBE_SCENARIO2_ON})
SCENARIO_2_FAMILY = "substantive-drinking"
SCENARIO_2_CHAT_ID = 920002
# Unchanged scenario #2 core prompt (product_contract_live core_scenarios[1]).
# Privacy: never logged; travels only through the production update seam.
SCENARIO_2_PROMPT = "К вечеру очень тянет выпить, как с этим обходиться?"

# kodmial/aa#553: validation-only real-book one-shot retrieval A/B. This is
# NOT a qualification mode and never changes the production flag default.
MODE_SCENARIO2 = "scenario2"
MODE_ONE_SHOT_AB = "one-shot-ab"
AB_SCHEMA_VERSION = "aa-one-shot-retrieval-ab/1"
AB_ISSUE = 553
AB_ARM_OFF = "off"
AB_ARM_ON = "on"
AB_STATUS_COMPLETE = "COMPLETE"
AB_STATUS_INCOMPLETE = "INCOMPLETE"
AB_STATUS_INCONCLUSIVE = "INCONCLUSIVE"
AB_STATUS_BLOCKED = "BLOCKED"
AB_STATUS_DEDUPLICATED = "DEDUPLICATED"
# Existing bounded canary budget (gate_c_canary): 300 s live, 600 s preflight
# and the 20 s in-budget cleanup reserve. The A/B runs two sequential arms, so
# its total real-model budget can never exceed two live budgets (600 s).
DEFAULT_AB_LIVE_BUDGET_S = 300.0
DEFAULT_AB_PREFLIGHT_BUDGET_S = 600.0
DEFAULT_AB_CLEANUP_RESERVE_S = 20.0
# A second arm is only started when enough job budget remains for a meaningful
# preflight plus the full live budget and cleanup reserve.
MIN_AB_PREFLIGHT_S = 60.0
ONE_SHOT_FASTPATH_ENV = "AA_RETRIEVAL_ONE_SHOT_FASTPATH"
AB_TRIGGER_FILE = "qualification/one_shot_retrieval_ab_553.trigger"
AB_MARKER_PREFIX = "<!-- aa-one-shot-retrieval-ab-553"

# kodmial/aa#424: campaign-level finite operational boundary (ONE ceiling for
# both probes; there is NO per-answer wall-clock cutoff). Mirrors the AA
# qualification's real-progress scenario allowance of 1800 s.
DEFAULT_CAMPAIGN_ALLOWANCE_S = 1800.0
# Lowest allowance that still permits meaningful observation plus artifact
# finalisation. When the remaining job budget is already inside the
# finalisation reserve, the allowance must not collapse to a token 1s.
MIN_CAMPAIGN_ALLOWANCE_S = 30.0
# Time kept for report/comment publication and artifact upload, reconciled
# against the Actions job budget when a job-start marker is provided.
DEFAULT_FINALISATION_RESERVE_S = 600.0
DEFAULT_SNAPSHOT_CADENCE_S = 15.0
DEFAULT_MAX_SNAPSHOTS = 480
DEFAULT_PROBE_A_PORT = "4097"
DEFAULT_PROBE_A_BOOTSTRAP_TIMEOUT_S = 60.0
DEFAULT_PROBE_B_BOOTSTRAP_TIMEOUT_S = 90.0
# Explicitly typed, answer-unrelated infrastructure bound: loopback TCP
# connect to the local OpenCode server. The model response has no total
# deadline; only the separate inactivity bound below applies after connect.
DEFAULT_PROBE_A_CONNECT_TIMEOUT_S = 10.0
# Mirror production ``HttpOpenCodeClient.request_timeout`` (AA_TRANSPORT_TIMEOUT_S):
# a per-socket-operation inactivity bound for model calls, never an
# end-to-end response deadline. A silent provider connection therefore
# cannot block forever, while a slow inference that keeps transferring
# bytes (or completes before the bound) still finishes normally.
DEFAULT_MODEL_INACTIVITY_TIMEOUT_S = 1800.0

SUPERVISOR_POLL_S = 0.25
STACK_DUMP_EVERY_S = 20.0
MAX_STACK_DUMPS = 16
COMMENT_CADENCE_S = 30.0
DEFAULT_TERMINATE_GRACE_S = 5.0
# Sustained local-server non-responsiveness (genuinely nonresponsive, proven
# by consecutive failed health probes) that allows typed cleanup instead of
# waiting for the campaign allowance. Local-infrastructure evidence only —
# never provider evidence.
NONRESPONSIVE_SNAPSHOTS_REQUIRED = 4
# A bounded loopback observation-probe bind (health/internal status only,
# never an answer/model call).
OBSERVATION_HTTP_TIMEOUT_S = 2.0
# Observation-only cadence while a model POST is in flight. This NEVER
# cancels or shortens the pending request; it records whether the local
# OpenCode server still answers bounded status probes.
MODEL_IN_FLIGHT_LIVENESS_INTERVAL_S = 15.0

# Diagnostic observability: continuity marker thresholds. Crossing these with
# the turn still undelivered is RECORDED, never enforced: the old 45/75/130
# answer windows are now evidence checkpoints instead of kill switches.
OBSERVATION_CONTINUITY_THRESHOLD_S: tuple[float, ...] = (45.0, 75.0, 130.0, 300.0, 900.0)
OBSERVATION_CONTINUITY_REPEAT_S = 900.0

# Internal child/supervisor contract only (diagnostic, never a Gate verdict):
# 0 delivered, 2 child bootstrap failed (root unproven),
# 3 inconclusive (unresolved/boundary-localized/stage-unobserved),
# 4 report could not be produced, 5 unexpected child exception,
# 6 proven provider error, 7 proven local server error.
EXIT_OK = 0
EXIT_BOOTSTRAP_FAILED = 2
EXIT_INCONCLUSIVE = 3
EXIT_REPORT_FAILED = 4
EXIT_CHILD_UNEXPECTED = 5
EXIT_PROVIDER_ERROR_PROVEN = 6
EXIT_SERVER_ERROR_PROVEN = 7

# Truthful classification labels (kodmial/aa#424 Definition of Done).
CLASS_DELIVERED = "delivered"
CLASS_PROVIDER_ERROR_PROVEN = "provider-error-proven"
CLASS_SERVER_ERROR_PROVEN = "server-error-proven"
CLASS_HARNESS_ERROR = "harness-error"
CLASS_UNRESOLVED = "unresolved-at-campaign-end"
CLASS_BOUNDARY_LOCALIZED = "boundary-localized-root-unknown"
CLASS_STAGE_UNOBSERVED = "stage-unobserved"

_EXIT_BY_CLASSIFICATION: dict[str, int] = {
    CLASS_DELIVERED: EXIT_OK,
    CLASS_PROVIDER_ERROR_PROVEN: EXIT_PROVIDER_ERROR_PROVEN,
    CLASS_SERVER_ERROR_PROVEN: EXIT_SERVER_ERROR_PROVEN,
    CLASS_HARNESS_ERROR: EXIT_CHILD_UNEXPECTED,
    CLASS_UNRESOLVED: EXIT_INCONCLUSIVE,
    CLASS_BOUNDARY_LOCALIZED: EXIT_INCONCLUSIVE,
    CLASS_STAGE_UNOBSERVED: EXIT_INCONCLUSIVE,
}

INCONCLUSIVE_CLASSIFICATIONS = frozenset(
    {CLASS_UNRESOLVED, CLASS_BOUNDARY_LOCALIZED, CLASS_STAGE_UNOBSERVED}
)

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
        "stage-end",
        "delivery",
        "scenario-start",
        "scenario-end",
        # Sanitized OpenCode internal status changes (session/message counts)
        # observed only when they CHANGE between snapshots: real provider-side
        # progress evidence, not a heartbeat.
        "opencode-internal",
    }
)

# Observed production semantic-DAG stage names. These name the OBSERVED
# boundary only; none of them (including ``opencode``) ever implies a proven
# provider root cause (kodmial/aa#424).
_STAGE_HINTS: dict[str, str] = {
    "opencode": (
        "Inspect the in-flight OpenCode send using the recorded session "
        "status/message part evidence: a pending response with zero "
        "provider events is UNOBSERVED, not proven provider failure."
    ),
    "provider": (
        "Inspect the in-flight model adapter send; provider failure must be "
        "proven from a real response payload before attribution."
    ),
    "planner": "Add bounded planner structured-output telemetry for the in-flight call.",
    "index": "Profile the RAM E5/FAISS index open captured in the snapshots.",
    "retrieval": "Profile the in-flight retrieval call and RRF/selection path.",
    "answer": "Inspect the in-flight answer synthesis call and its structured output handling.",
    "verifier": "Inspect the in-flight verifier turn and its structured output handling.",
    "whole_turn_judge": "Inspect the in-flight whole-turn judge call.",
    "finalize": "Inspect answer finalization/envelope handling at the observed stage.",
    "turn": "Inspect the LangGraph turn-boundary transitions in the snapshot stacks.",
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


def _install_stack_handler(
    path: Path,
    *,
    chain: bool = False,
    warn: Callable[[str], None] | None = None,
) -> bool:
    """Turn SIGUSR1 into a nonfatal whole-process faulthandler dump (append).

    ``chain=False`` is mandatory: chaining into the prior/default SIGUSR1
    disposition terminates the process right after the stack dump. That
    killed BOTH diagnostic probes at ~20s of child lifetime in the real
    run 39767853106 (both children exit code -10 == SIGUSR1), truncating
    the observation windows. The handler must ONLY dump stacks and return;
    the probe keeps running until it reaches its own typed end state.

    Returns ``True`` when the handler is registered. On failure the caller
    is warned (never silent) and the supervisor's readiness sentinel is NOT
    written, so the supervisor never SIGUSR1s an unregistered child (the
    default SIGUSR1 action would terminate it).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a", encoding="utf-8")
        faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True, chain=chain)
        _HELD_STACK_STREAMS.append(stream)
        return True
    except (OSError, ValueError) as exc:
        if warn is not None:
            try:
                warn("signal-handler-registration-failed:" + type(exc).__name__)
            except Exception:
                pass
        return False


_HELD_STACK_STREAMS: list[Any] = []


def _stack_handler_ready_path(out_dir: Path, probe: str) -> Path:
    """Sentinel file proving the child's SIGUSR1 handler is registered."""
    return out_dir / f"stack-handler-ready-{probe}.txt"


def _summarize_messages(payload: Any) -> tuple[int | None, list[str]]:
    """Sanitized message-list summary: counts + part kinds (no text, no ids)."""
    if not isinstance(payload, list):
        return None, []
    kinds: list[str] = []
    for message in payload:
        if not isinstance(message, dict):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict):
                kind = part.get("type")
                if isinstance(kind, str) and len(kinds) < 8 and kind not in kinds:
                    kinds.append(kind[:24])
    return len(payload), kinds


def _opencode_internal_observation(
    base_url: str, session_id: str, *, timeout: float = OBSERVATION_HTTP_TIMEOUT_S
) -> dict[str, Any]:
    """Sanitized OpenCode internal status snapshot (bounded, fail-soft).

    kodmial/aa#424: discriminate live inference from server HTTP-wait by
    observing the pinned OpenCode server's own state when supported:
    sessions count and (when the diagnostic owns a session id) the
    pending assistant message part kinds. Counts and part kinds only —
    never message text, never session identifiers.
    """
    result: dict[str, Any] = {"responsive": False, "sessions_endpoint_supported": None}
    ok, _provider_error, timings, extra = _timed_request(
        base_url, "GET", "/session", None, timeout, timeout
    )
    result["responsive"] = ok
    result["sessions_endpoint_supported"] = ok
    if not ok:
        result["fail_category"] = str(timings.get("fail_category", "http-error"))[:48]
        return result
    sessions = extra.get("_list")
    result["sessions_seen"] = len(sessions) if isinstance(sessions, list) else None
    if not session_id:
        return result
    ok_messages, _err, mtimings, mextra = _timed_request(
        base_url,
        "GET",
        f"/session/{urllib.parse.quote(session_id)}/message",
        None,
        timeout,
        timeout,
    )
    if ok_messages:
        result["messages_endpoint_supported"] = True
        result["messages_seen"] = mextra.get("_messages_seen")
        result["part_kinds"] = mextra.get("_part_kinds")
    else:
        result["messages_endpoint_supported"] = False
        result["messages_fail_category"] = str(mtimings.get("fail_category", ""))[:48]
    return result


class ChildWatchdog(threading.Thread):
    """Independent in-child snapshot thread (survives event-loop wedges).

    kodmial/aa#424: fixed 10/25/45/75/110 s offsets were replaced by a
    bounded periodic cadence, so the whole observation window receives
    incremental snapshots up to the campaign allowance while keeping
    bounded log volume. Each snapshot records full in-process state, the
    OpenCode request audit and sanitized internal server status;
    whole-process faulthandler stacks are separately requested by the
    supervisor via SIGUSR1.
    """

    _MAX_THREADS = 12
    _MAX_FRAMES = 8
    _MAX_TASKS = 12

    def __init__(
        self,
        *,
        probe: str,
        cadence_s: float,
        max_snapshots: int,
        anchor: threading.Event,
        anchor_wait_timeout_s: float,
        sink: EventSink,
        checkpoint_dir: Path,
        opencode_base_url: str,
        counters: dict[str, Any],
        audit_provider: Callable[[], list[dict[str, Any]]] | None = None,
        internal_probe: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(name=f"scenario2-diag-watchdog-{probe}", daemon=True)
        self.probe = probe
        self.cadence_s = max(0.05, float(cadence_s))
        self.max_snapshots = max(1, int(max_snapshots))
        self.anchor = anchor
        self.anchor_wait_timeout_s = max(1.0, float(anchor_wait_timeout_s))
        self.sink = sink
        self.checkpoint_dir = checkpoint_dir
        self.opencode_base_url = opencode_base_url
        self.counters = counters
        self.audit_provider = audit_provider
        self.internal_probe = internal_probe
        self.main_loop: asyncio.AbstractEventLoop | None = None
        self._stop_requested = threading.Event()
        self._consecutive_health_failures = 0
        self._last_internal: dict[str, Any] = {"_baseline": None}

    def run(self) -> None:
        if not self.anchor.wait(timeout=self.anchor_wait_timeout_s):
            return
        start = time.monotonic()
        for index in range(1, self.max_snapshots + 1):
            delay = start + index * self.cadence_s - time.monotonic()
            if delay > 0 and self._stop_requested.wait(timeout=delay):
                return
            if self._stop_requested.is_set() or not self.anchor.is_set():
                return
            self._snapshot(index=index, at_s=round(time.monotonic() - start, 2))

    def stop(self) -> None:
        self._stop_requested.set()

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
        tasks = self._collect_async_tasks()
        health_ok: Any = None
        health_elapsed_ms: Any = None
        try:
            probe_start = time.monotonic()
            health_ok, _ = _health_probe(self.opencode_base_url, OBSERVATION_HTTP_TIMEOUT_S)
            health_elapsed_ms = round((time.monotonic() - probe_start) * 1000.0, 1)
        except Exception:
            health_ok = None
        if health_ok:
            self._consecutive_health_failures = 0
            self.counters["last_health_ok_at_s"] = round(time.monotonic(), 2)
        else:
            self._consecutive_health_failures += 1
            if self._consecutive_health_failures >= NONRESPONSIVE_SNAPSHOTS_REQUIRED:
                # Genuinely nonresponsive LOCAL server (proven local
                # infrastructure failure, never provider evidence).
                self.counters["server_nonresponsive_sustained"] = True
                try:
                    self.sink.emit(
                        "server-nonresponsive-sustained",
                        consecutive_failures=self._consecutive_health_failures,
                        last_health_ok_at_s=self.counters.get("last_health_ok_at_s"),
                    )
                except Exception:
                    pass
        last_stage_mono = self.counters.get("last_stage_monotonic")
        seconds_since_last_stage = (
            round(time.monotonic() - float(last_stage_mono), 2)
            if isinstance(last_stage_mono, (int, float))
            else None
        )
        last_real_mono = self.counters.get("last_real_monotonic")
        seconds_since_last_real = (
            round(time.monotonic() - float(last_real_mono), 2)
            if isinstance(last_real_mono, (int, float))
            else None
        )
        internal = self._internal_observation()
        return {
            "probe": self.probe,
            "at_s": at_s,
            "stage": str(self.counters.get("last_stage", "") or "")[:64],
            "stage_at_s": self.counters.get("last_stage_at_s"),
            "seconds_since_last_stage": seconds_since_last_stage,
            "last_real_event": str(self.counters.get("last_real_event", "") or "")[:64],
            "seconds_since_last_real": seconds_since_last_real,
            "typing_sends": int(self.counters.get("typing_sends", 0)),
            "delivered": bool(self.counters.get("delivered", False)),
            "consecutive_health_failures": self._consecutive_health_failures,
            "threads": threads,
            "asyncio_tasks": tasks,
            "opencode_health": {"ok": health_ok, "elapsed_ms": health_elapsed_ms},
            "opencode_internal": internal,
            "opencode_request_audit": self._audit_summary(),
            "sockets": summarize_loopback_sockets(),
            "opencode_processes": count_opencode_processes(),
        }

    def _internal_observation(self) -> dict[str, Any]:
        """Periodic sanitized OpenCode internal status (fail-soft, bounded)."""
        if self.internal_probe is None:
            return {}
        try:
            internal = self.internal_probe()
        except Exception:
            return {"responsive": False}
        internal = internal if isinstance(internal, dict) else {}
        baseline = (internal.get("sessions_seen"), internal.get("messages_seen"))
        if self._last_internal.get("_baseline") != baseline:
            # Actual provider-side work moved (session/message counts
            # changed): genuine progress evidence, never a heartbeat.
            self._last_internal["_baseline"] = baseline
            self.counters["last_real_event"] = "opencode-internal"
            self.counters["last_real_monotonic"] = time.monotonic()
        return internal

    def _collect_async_tasks(self) -> list[dict[str, Any]]:
        """Inspect tasks from the event-loop thread (never from this thread).

        ``asyncio.all_tasks`` is only safe inside the loop thread. The
        watchdog therefore schedules the read with ``call_soon_threadsafe``
        and waits for a bounded result; a wedged loop simply yields an empty
        list instead of touching loop state from the background thread.
        """
        loop = self.main_loop
        if loop is None:
            return []
        future: concurrent.futures.Future[list[dict[str, Any]]] = concurrent.futures.Future()

        def collect() -> None:
            if future.cancelled():
                return
            try:
                pending = list(asyncio.all_tasks(loop))[: self._MAX_TASKS]
            except Exception:
                pending = []
            result: list[dict[str, Any]] = []
            for task in pending:
                try:
                    label = sanitize_frame(task.get_coro().cr_code)
                except Exception:
                    label = "<opaque-task>"
                result.append({"name": task.get_name()[:48], "coro": label, "done": task.done()})
            try:
                future.set_result(result)
            except Exception:
                pass

        try:
            loop.call_soon_threadsafe(collect)
        except RuntimeError:
            return []
        try:
            return future.result(timeout=0.5)
        except TimeoutError:
            future.cancel()
            return []

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
    """Public alias so the diagnostic owns one health implementation."""
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


def _model_inactivity_timeout_s() -> float:
    """Production-equivalent socket inactivity bound for model calls.

    Mirrors ``HttpOpenCodeClient.request_timeout`` (``AA_TRANSPORT_TIMEOUT_S``).
    This is a per-socket-operation inactivity bound, not an end-to-end
    response deadline: a genuinely slow provider that keeps the connection
    active is not failed merely for exceeding a wall-clock window.
    """
    try:
        from aa.time_policy import current_time_policy

        value = float(current_time_policy().transport_timeout_s)
    except Exception:
        value = DEFAULT_MODEL_INACTIVITY_TIMEOUT_S
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_MODEL_INACTIVITY_TIMEOUT_S
    return value


def _timed_request(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, object] | None,
    connect_timeout: float,
    op_timeout: float | None,
) -> tuple[bool, bool, dict[str, float], dict[str, Any]]:
    """One instrumented loopback HTTP call with connect/write/read phases.

    kodmial/aa#424: the connect bound covers ONLY the TCP establishment
    phase of the loopback infrastructure. The read phase has a finite
    per-socket inactivity bound (production ``HttpOpenCodeClient``
    ``request_timeout`` semantics), but there is NO total-response deadline:
    a healthy slow inference that keeps the connection alive is never
    falsely failed merely for taking long. A provider connection that
    accepts TCP but never sends a byte therefore times out with a typed
    ``read-timeout`` instead of blocking until the campaign allowance.
    Response payloads are never returned or logged. The returned ``extra``
    may hold internal opaque values (session id reference, sanitized
    counts) for the very next request; identifiers are never emitted.
    """
    try:
        host, port = http_split_probe_base_url(base_url)
    except ValueError:
        return False, False, {"fail_category": "bad-base-url"}, {}
    timings: dict[str, float] = {}
    extra: dict[str, Any] = {}
    connection: http.client.HTTPConnection | None = None
    phase = "connect"
    try:
        start = time.perf_counter()
        connection = http.client.HTTPConnection(host, port, timeout=connect_timeout)
        connection.connect()
        timings["connect_ms"] = round((time.perf_counter() - start) * 1000.0, 1)
        inactivity_timeout = (
            float(op_timeout) if op_timeout is not None else _model_inactivity_timeout_s()
        )
        if connection.sock is not None:
            connection.sock.settimeout(inactivity_timeout)
            timings["inactivity_timeout_s"] = round(inactivity_timeout, 1)
        phase = "write"
        write_start = time.perf_counter()
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=payload, headers=headers)
        timings["write_ms"] = round((time.perf_counter() - write_start) * 1000.0, 1)
        phase = "read"
        read_start = time.perf_counter()
        response = connection.getresponse()
        raw = response.read()
        timings["read_ms"] = round((time.perf_counter() - read_start) * 1000.0, 1)
        timings["status"] = int(response.status)
        ok = response.status == 200
        if ok:
            if method == "POST" and path == "/session":
                session = json.loads(raw.decode("utf-8"))
                session_id = session.get("id")
                if isinstance(session_id, str) and session_id:
                    extra["_id"] = session_id
            elif method == "GET" and path == "/session":
                listed = json.loads(raw.decode("utf-8"))
                extra["_list"] = listed if isinstance(listed, list) else None
            elif method == "GET" and path.endswith("/message"):
                listed = json.loads(raw.decode("utf-8"))
                extra["_messages_seen"], extra["_part_kinds"] = _summarize_messages(listed)
            decision = _classify_opencode_payload(raw)
            extra.update(decision)
            return ok, decision.get("provider_error", False), timings, extra
        extra.update({"fail_category": f"http-error-{timings.get('status', '?')}"})
        return ok, False, timings, extra
    except (TimeoutError, OSError) as exc:
        timings["fail_category"] = (
            f"{phase}-timeout" if isinstance(exc, TimeoutError) else "network-error"
        )
        return False, False, timings, extra
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


async def _timed_request_async(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, object] | None,
    connect_timeout: float,
    op_timeout: float | None,
) -> tuple[bool, bool, dict[str, float], dict[str, Any]]:
    """Run ``_timed_request`` on a DAEMON thread (bounded child lifetime).

    The default asyncio executor joins non-daemon threads at interpreter
    shutdown, so a hung unbounded response could leak the child past the
    campaign boundary. A dedicated daemon thread lets the child exit
    normally right after bounded cleanup while the blocked socket dies
    with the process.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[tuple[bool, bool, dict[str, float], dict[str, Any]]] = (
        loop.create_future()
    )

    def deliver(ok: bool, value: Any) -> None:
        if future.done():
            return
        if ok:
            future.set_result(value)
        else:
            future.set_exception(value)

    def runner() -> None:
        try:
            outcome = _timed_request(base_url, method, path, body, connect_timeout, op_timeout)
        except BaseException as exc:  # noqa: BLE001 (fail-soft delivery)
            try:
                loop.call_soon_threadsafe(deliver, False, exc)
            except RuntimeError:
                pass
            return
        try:
            loop.call_soon_threadsafe(deliver, True, outcome)
        except RuntimeError:
            pass

    threading.Thread(target=runner, name="diag-timed-request", daemon=True).start()
    return await future


def _observe_model_in_flight_liveness(
    *,
    sink: EventSink,
    base_url: str,
    session_id_provider: Callable[[], str],
    started_monotonic: float,
    stop_event: threading.Event,
    interval_s: float = MODEL_IN_FLIGHT_LIVENESS_INTERVAL_S,
) -> None:
    """Observe local-server liveness while a model POST is pending.

    Observation-only: this thread never cancels, times out or otherwise
    changes the in-flight request. It periodically proves whether the local
    OpenCode server (and, when available, the session/message status
    endpoints) still answers bounded probes, so a wedged local server is
    distinguishable from a slow provider. A provider that accepts TCP but
    sends no bytes remains unobservable here and is the documented
    campaign-allowance-bounded accepted risk.
    """
    interval = max(1.0, float(interval_s))
    while not stop_event.wait(interval):
        try:
            health_ok, _ = _health_probe(base_url, OBSERVATION_HTTP_TIMEOUT_S)
            internal = _opencode_internal_observation(
                base_url,
                str(session_id_provider() or ""),
                timeout=OBSERVATION_HTTP_TIMEOUT_S,
            )
            sink.emit(
                "model-in-flight-liveness",
                elapsed_ms=round((time.monotonic() - started_monotonic) * 1000.0, 1),
                server_healthy=bool(health_ok),
                sessions_responsive=bool(internal.get("responsive")),
                sessions_seen=internal.get("sessions_seen"),
                messages_seen=internal.get("messages_seen"),
                part_kinds=internal.get("part_kinds"),
            )
        except Exception:
            return


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


async def _bounded_async_stop(item: Any, timeout_s: float) -> bool:
    """Bounded best-effort shutdown of runtime/app (never blocks forever)."""
    try:
        await asyncio.wait_for(item.stop(), timeout=max(1.0, float(timeout_s)))
        return True
    except Exception:
        return False


def _forced_bounded_exit(sink: EventSink, probe: str, exit_code: int) -> None:
    """Terminate the child probe process group AFTER evidence is flushed.

    Required when graceful teardown timed out because the production-path
    resources could not be released: normal interpreter shutdown joins the
    executor threads, so a stuck drain would leak the child past the
    campaign boundary. This documented bounded end targets the CHILD'S OWN
    process group — ``start_new_session=True`` makes the child a session
    leader, so this includes the spawned OpenCode server and never the
    supervisor/parent group. Event ids are already on disk and in the
    Actions log.
    """
    sink.emit("child-forced-exit", probe=probe, exit_code=int(exit_code))
    sink.close()
    sys.stdout.flush()
    try:
        own_pgid = os.getpgid(os.getpid())
    except OSError:
        own_pgid = None
    if own_pgid is not None:
        try:
            os.killpg(own_pgid, signal.SIGKILL)
            return
        except OSError:
            pass
    os._exit(int(exit_code))


async def run_probe_a(argv: dict[str, Any], sink: EventSink, counters: dict[str, Any]) -> int:
    """Probe A child: health, session create, one message on the AA model.

    kodmial/aa#424: the model message has NO wall-clock cutoff (production
    ``HttpOpenCodeClient`` semantics: only a per-socket inactivity bound,
    no total-response deadline); a real call of 46/75/300/900 s completes
    and is recorded with its timing.
    """
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
    bootstrap_timeout = float(
        argv.get("probe_a_bootstrap_timeout_s", DEFAULT_PROBE_A_BOOTSTRAP_TIMEOUT_S)
    )
    connect_timeout = float(argv.get("connect_timeout_s", DEFAULT_PROBE_A_CONNECT_TIMEOUT_S))
    model_inactivity_timeout = _model_inactivity_timeout_s()
    session_state = {"id": ""}
    anchor = threading.Event()
    watchdog = ChildWatchdog(
        probe=PROBE_DIRECT_MODEL,
        cadence_s=float(argv.get("snapshot_cadence_s", DEFAULT_SNAPSHOT_CADENCE_S)),
        max_snapshots=int(argv.get("max_snapshots", DEFAULT_MAX_SNAPSHOTS)),
        anchor=anchor,
        anchor_wait_timeout_s=max(90.0, bootstrap_timeout + 30.0),
        sink=sink,
        checkpoint_dir=Path(argv["out_dir"]),
        opencode_base_url=base_url,
        counters=counters,
        internal_probe=lambda: _opencode_internal_observation(base_url, session_state["id"]),
    )
    watchdog.main_loop = asyncio.get_running_loop()
    watchdog.start()
    sink.emit("probe-start", probe=PROBE_DIRECT_MODEL, model=str(model)[:96])
    counters["last_real_event"] = "probe-start"
    counters["last_real_monotonic"] = time.monotonic()
    exit_code = EXIT_OK
    session_id = ""
    try:
        # 1) Pre-probe: attach vs spawn evidence (production startup path).
        attached, _ = await asyncio.to_thread(_health_probe, base_url, 1.0)
        started = time.perf_counter()
        try:
            await asyncio.wait_for(runtime.start(), timeout=bootstrap_timeout)
            health_ok = bool(runtime.running and runtime.ready)
            await asyncio.wait_for(runtime.ensure_ready(5.0), timeout=bootstrap_timeout)
        except Exception as exc:
            sink.emit(
                "server-health",
                ok=False,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
                attached=attached,
                fail_category=type(exc).__name__[:48],
            )
            sink.emit("probe-end", probe=PROBE_DIRECT_MODEL, status=CLASS_SERVER_ERROR_PROVEN)
            return EXIT_SERVER_ERROR_PROVEN
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
            sink.emit("probe-end", probe=PROBE_DIRECT_MODEL, status=CLASS_SERVER_ERROR_PROVEN)
            return EXIT_SERVER_ERROR_PROVEN
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
        ok, _provider_error, timings, extra = await _timed_request_async(
            base_url,
            "POST",
            "/session",
            {"title": "aa-scenario2-diagnostic"},
            connect_timeout,
            10.0,
        )
        session_id = str(extra.get("_id", "")) if ok else ""
        session_state["id"] = session_id
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
            # Session creation is bounded startup infrastructure; a
            # transport failure here is proven local-server evidence.
            sink.emit(
                "probe-end",
                probe=PROBE_DIRECT_MODEL,
                status=CLASS_SERVER_ERROR_PROVEN,
                fail_category=str(timings.get("fail_category", "") or "session-create-failed")[:48],
            )
            return EXIT_SERVER_ERROR_PROVEN

        # 3) One real model message with NO wall-clock cutoff: the TCP
        #    connect phase and per-socket inactivity are bounded (typed
        #    infrastructure bounds), never total response time. A real
        #    response is awaited on a daemon thread and recorded with full
        #    phase timing.
        sink.emit(
            "model-send-start",
            probe=PROBE_DIRECT_MODEL,
            model=str(model)[:96],
            read_inactivity_s=round(model_inactivity_timeout, 1),
        )
        counters["last_real_event"] = "model-send-start"
        counters["last_real_monotonic"] = time.monotonic()
        anchor.set()
        started = time.perf_counter()
        liveness_stop = threading.Event()
        liveness = threading.Thread(
            target=_observe_model_in_flight_liveness,
            kwargs={
                "sink": sink,
                "base_url": base_url,
                "session_id_provider": lambda: session_state["id"],
                "started_monotonic": time.monotonic(),
                "stop_event": liveness_stop,
            },
            name="diag-model-in-flight-liveness",
            daemon=True,
        )
        liveness.start()
        try:
            ok, provider_error, timings, _extra = await _timed_request_async(
                base_url,
                "POST",
                f"/session/{urllib.parse.quote(session_id)}/message",
                _probe_a_message_body(model),
                connect_timeout,
                model_inactivity_timeout,
            )
        finally:
            liveness_stop.set()
            liveness.join(timeout=2.0)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        fail_category = str(timings.get("fail_category", "") or "")
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
        elif provider_error:
            # PROVEN from a real error response payload.
            sink.emit(
                "probe-end",
                probe=PROBE_DIRECT_MODEL,
                status=CLASS_PROVIDER_ERROR_PROVEN,
                fail_category=fail_category,
            )
            exit_code = EXIT_PROVIDER_ERROR_PROVEN
        else:
            # Transport-level failure against the LOCAL server: proven
            # local infrastructure evidence, never provider blame.
            sink.emit(
                "probe-end",
                probe=PROBE_DIRECT_MODEL,
                status=CLASS_SERVER_ERROR_PROVEN,
                fail_category=fail_category or "model-request-failed",
            )
            exit_code = EXIT_SERVER_ERROR_PROVEN
        return exit_code
    finally:
        # Bounded best-effort cleanup; the production runtime reaps its child.
        if session_id:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        _timed_request,
                        base_url,
                        "POST",
                        f"/session/{urllib.parse.quote(session_id)}/abort",
                        None,
                        connect_timeout,
                        5.0,
                    ),
                    timeout=6.0,
                )
            except Exception:
                pass
            try:
                await asyncio.wait_for(
                    _timed_request_async(
                        base_url,
                        "DELETE",
                        f"/session/{urllib.parse.quote(session_id)}",
                        None,
                        connect_timeout,
                        5.0,
                    ),
                    timeout=7.0,
                )
            except Exception:
                pass
            session_id = ""
            session_state["id"] = ""
        stopped = await _bounded_async_stop(runtime, 10.0)
        anchor.set()
        sink.emit("child-exit", probe=PROBE_DIRECT_MODEL, exit_code=exit_code, cleanup_done=stopped)
        watchdog.stop()
        if not stopped:
            _forced_bounded_exit(sink, PROBE_DIRECT_MODEL, exit_code)


# ---------------------------------------------------------------------------
# Probe B (scenario2): the exact production scenario #2 turn
# ---------------------------------------------------------------------------


class _DiagnosticTelegramApi:
    """Deterministic in-process Telegram network seam (no external network).

    Minimal standalone diagnostic mirror of the qualification lane seam:
    the production ``PollingTelegramTransport`` stays entirely real and
    only this Bot API surface is replaced by an in-process recorder.
    Polling volume is bounded in memory and frequency (kodmial/aa#424).
    """

    _MAX_RECORDED_CALLS = 500

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.get_updates_count = 0
        self.sent_texts: list[str] = []
        self.chat_actions = 0
        self.pending_updates: list[dict[str, Any]] = []
        self._counter = 0

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        if method == "getUpdates":
            self.get_updates_count += 1
        elif len(self.calls) < self._MAX_RECORDED_CALLS:
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
            await asyncio.sleep(0.05)
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


def _emit_observation_continuity(
    sink: EventSink,
    counters: dict[str, Any],
    turn_started: float,
    state: dict[str, Any],
) -> None:
    """Record that observation continues past legacy answer windows.

    kodmial/aa#424: crossing 45/75/130/300/900 s is pure evidence: the
    turn is still observed and continues — no cutoff exists anywhere.
    After the named thresholds the markers repeat at a bounded cadence.
    """
    elapsed = time.monotonic() - turn_started
    index = int(state.get("index", 0))
    thresholds = OBSERVATION_CONTINUITY_THRESHOLD_S
    if index < len(thresholds):
        if elapsed < thresholds[index]:
            return
        threshold = thresholds[index]
        state["index"] = index + 1
        state["next_at"] = elapsed + OBSERVATION_CONTINUITY_REPEAT_S
    else:
        if elapsed < float(state.get("next_at", 0.0)):
            return
        threshold = elapsed
        state["next_at"] = elapsed + OBSERVATION_CONTINUITY_REPEAT_S
    last_real_mono = counters.get("last_real_monotonic")
    sink.emit(
        "observation-continues",
        elapsed_s=round(elapsed, 1),
        threshold_s=round(float(threshold), 1),
        last_stage=str(counters.get("last_stage", "") or "")[:64],
        last_real_event=str(counters.get("last_real_event", "") or "")[:64],
        seconds_since_last_real=(
            round(time.monotonic() - float(last_real_mono), 1)
            if isinstance(last_real_mono, (int, float))
            else None
        ),
        delivered=bool(counters.get("delivered", False)),
    )


# ---------------------------------------------------------------------------
# kodmial/aa#553: bounded real-book one-shot retrieval A/B helpers
# ---------------------------------------------------------------------------


def _ab_canonical_scenario() -> tuple[str, int, str, str]:
    """The frozen substantive scenario and its provenance (never emitted raw).

    Prefer the exact production definition in ``GATE_C_CANARY_SCENARIOS`` so
    the A/B can never drift into a separate prompt. The local constant is the
    fail-closed fallback when the qualification module is unavailable.
    """
    try:
        from aa.qualification.product_contract_live import GATE_C_CANARY_SCENARIOS

        for family, chat_id, prompt in GATE_C_CANARY_SCENARIOS:
            if family == SCENARIO_2_FAMILY:
                return str(family), int(chat_id), str(prompt), "GATE_C_CANARY_SCENARIOS"
    except Exception:
        pass
    return SCENARIO_2_FAMILY, SCENARIO_2_CHAT_ID, SCENARIO_2_PROMPT, "scenario2-constant"


def ab_scenario_identity() -> dict[str, Any]:
    """Privacy-safe identity of the frozen scenario (digest, never text)."""
    family, chat_id, prompt, source = _ab_canonical_scenario()
    return {
        "family": family,
        "chat_id": chat_id,
        "scenario_index": 2,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "source": source,
    }


def ab_config_identity(
    primary_model: str,
    fallback_model: str,
    *,
    live_budget_s: float = DEFAULT_AB_LIVE_BUDGET_S,
    preflight_budget_s: float = DEFAULT_AB_PREFLIGHT_BUDGET_S,
    cleanup_reserve_s: float = DEFAULT_AB_CLEANUP_RESERVE_S,
) -> dict[str, Any]:
    """Identity of one A/B configuration for dedupe (no run id, no secrets)."""
    scenario = ab_scenario_identity()
    payload = {
        "schema": AB_SCHEMA_VERSION,
        "scenario": scenario,
        "primary_model": str(primary_model or ""),
        "fallback_model": str(fallback_model or ""),
        "arms": [AB_ARM_OFF, AB_ARM_ON],
        "live_budget_s": round(float(live_budget_s), 3),
        "preflight_budget_s": round(float(preflight_budget_s), 3),
        "cleanup_reserve_s": round(float(cleanup_reserve_s), 3),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return {**payload, "config_digest": digest}


def ab_run_marker(run_id: str) -> str:
    """Idempotence marker for one A/B run comment."""
    return f"{AB_MARKER_PREFIX} run={run_id} -->"


def ab_complete_marker(main_sha: str, config_digest: str) -> str:
    """Durable marker for one COMPLETE A/B result (same SHA + config)."""
    return (
        f"{AB_MARKER_PREFIX} sha={main_sha} config={config_digest} status={AB_STATUS_COMPLETE} -->"
    )


def expected_ab_env_flag(arm: str) -> str:
    """The one explicit env value each A/B arm must observe (0=OFF, 1=ON)."""
    return "1" if str(arm) == AB_ARM_ON else "0"


def ab_arm_environment(arm: str) -> dict[str, str]:
    """Isolated per-child environment for one A/B arm.

    The production flag is never set in the parent process; each arm child
    receives exactly one explicit value so the two arms cannot share state.
    """
    if str(arm) not in (AB_ARM_OFF, AB_ARM_ON):
        raise ValueError("unknown A/B arm")
    return {ONE_SHOT_FASTPATH_ENV: expected_ab_env_flag(str(arm))}


def _ab_budget(
    live_budget_s: float = DEFAULT_AB_LIVE_BUDGET_S,
    preflight_budget_s: float = DEFAULT_AB_PREFLIGHT_BUDGET_S,
    cleanup_reserve_s: float = DEFAULT_AB_CLEANUP_RESERVE_S,
) -> dict[str, float]:
    """Per-arm bounded canary budget (300 s live / 600 s preflight / reserve)."""
    live = max(1.0, float(live_budget_s))
    preflight = max(1.0, float(preflight_budget_s))
    reserve = max(0.0, min(float(cleanup_reserve_s), live / 4.0))
    return {
        "live_budget_s": round(live, 3),
        "preflight_budget_s": round(preflight, 3),
        "cleanup_reserve_s": round(reserve, 3),
        "live_model_budget_s": round(max(1.0, live - reserve), 3),
        "wall_budget_s": round(preflight + live, 3),
        "total_real_model_budget_cap_s": round(2.0 * live, 3),
    }


def _install_ab_retrieval_probe() -> tuple[dict[str, Any], Any]:
    """Capture the real retrieval call path without touching ``src/``.

    ``retrieval_node`` resolves ``aretrieve_with_semantic_selection`` as a
    module global at call time, so wrapping that global observes the actual
    production retrieval invocation and its sanitized metadata.
    """
    from aa.conversation import retrieval_node

    original = retrieval_node.aretrieve_with_semantic_selection
    capture: dict[str, Any] = {"calls": 0, "metadata": {}, "error_type": ""}

    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        capture["calls"] = int(capture.get("calls", 0)) + 1
        try:
            pack = await original(*args, **kwargs)
        except Exception as exc:
            capture["error_type"] = type(exc).__name__[:64]
            raise
        try:
            metadata = dict(getattr(pack, "retrieval_metadata", {}) or {})
            capture["metadata"] = retrieval_node.sanitize_retrieval_metadata(metadata)
        except Exception:
            capture["metadata"] = {}
        return pack

    retrieval_node.aretrieve_with_semantic_selection = _wrapped
    return capture, original


def _restore_ab_retrieval_probe(original: Any) -> None:
    """Restore the production retrieval entry point after one A/B arm."""
    try:
        from aa.conversation import retrieval_node

        retrieval_node.aretrieve_with_semantic_selection = original
    except Exception:
        pass


def _ab_retrieval_evidence(capture: dict[str, Any] | None) -> dict[str, Any]:
    """Privacy-safe observed retrieval path and model-call accounting."""
    capture = capture if isinstance(capture, dict) else {}
    metadata = capture.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    safe: dict[str, Any] = {
        "calls_observed": int(capture.get("calls", 0) or 0),
        "metadata_present": bool(metadata),
        "error_type": str(capture.get("error_type", ""))[:64],
    }
    for key in (
        "retrieval_mode",
        "fastpath_attempted",
        "fallback_reason",
        "coverage_status",
        "LLM_call_count",
        "selector_ms",
        "coverage_ms",
        "total_ms",
        "loop_model_calls",
        "loop_iterations",
        "loop_searches",
        "provider_call_count",
        "provider_p50_ms",
        "provider_p95_ms",
        "provider_max_ms",
        "turn_elapsed_ms",
        "turn_remaining_ms",
        "local_retrieval_ms",
        "typed_stage",
        "retrieval_backend",
        "selection_route",
        "selected_passages",
        "total_tokens",
    ):
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
    durations = metadata.get("provider_call_durations_ms")
    if isinstance(durations, list):
        safe["provider_call_durations_ms_count"] = len(durations)
    budget_snapshot = metadata.get("turn_budget_snapshot")
    if isinstance(budget_snapshot, dict):
        stages = [
            str(item.get("stage", ""))[:64]
            for item in list(budget_snapshot.get("stages", []) or [])
            if isinstance(item, dict)
        ]
        safe["budget_stages"] = stages[:32]
        safe["provider_429"] = "provider_429" in stages
        safe["provider_semantic_timeout"] = "provider_semantic_timeout" in stages
    return safe


def _ab_attempt_summary(
    attempts: Any,
    request_audit: Any = None,
) -> dict[str, Any]:
    """Observed model HTTP attempts by logical agent (counts only)."""
    by_agent: dict[str, dict[str, Any]] = {}
    failure_categories: list[str] = []
    observed_models: list[str] = []
    total = 0
    provider_429 = False
    active_backoff = False
    entries = list(attempts) if isinstance(attempts, (list, tuple)) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        agent = str(entry.get("audit_agent") or entry.get("agent") or "")[:64]
        if not agent:
            continue
        total += 1
        bucket = by_agent.setdefault(
            agent,
            {
                "attempts": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "in_flight": 0,
                "latency_ms_total": 0.0,
            },
        )
        bucket["attempts"] = int(bucket["attempts"]) + 1
        state = str(entry.get("state", ""))[:32]
        if state in ("completed", "failed", "cancelled", "in_flight"):
            bucket[state] = int(bucket[state]) + 1
        latency = entry.get("latency_ms")
        if isinstance(latency, (int, float)):
            bucket["latency_ms_total"] = round(
                float(bucket["latency_ms_total"]) + float(latency), 1
            )
        failure = str(entry.get("failure_category", "") or "")[:64]
        if failure:
            failure_categories.append(failure)
        model = str(entry.get("model", "") or "")[:96]
        fallback = str(entry.get("fallback_model", "") or "")[:96]
        for value in (model, fallback):
            if value and value not in observed_models:
                observed_models.append(value)
    rows = list(request_audit) if isinstance(request_audit, (list, tuple)) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        failure = str(row.get("failure_category", "") or "")[:64]
        error_kind = str(row.get("error_kind", "") or "")[:64]
        combined = (failure + " " + error_kind).casefold()
        if failure and failure not in failure_categories:
            failure_categories.append(failure)
        if "429" in combined or "ratelimit" in combined or "rate_limit" in combined:
            provider_429 = True
        if "backoff" in combined or "retry" in combined:
            active_backoff = True
    for failure in failure_categories:
        folded = failure.casefold()
        if "429" in folded or "ratelimit" in folded or "rate_limit" in folded:
            provider_429 = True
        if "backoff" in folded:
            active_backoff = True
    return {
        "present": bool(entries),
        "model_http_attempts": total,
        "by_agent": by_agent,
        "failure_categories": failure_categories[:16],
        "observed_models": observed_models[:8],
        "request_audit_present": bool(rows),
        "provider_429": provider_429,
        "active_backoff": active_backoff,
    }


_AB_TELEMETRY_KEYS: tuple[str, ...] = (
    "planner_query_count",
    "planner_outcome",
    "planner_reason",
    "planner_latency_ms",
    "retrieval_passages",
    "retrieval_hits",
    "retrieval_outcome",
    "retrieval_latency_ms",
    "retrieval_over_budget",
    "answer_outcome",
    "answer_latency_ms",
    "answer_rounds",
    "verifier_outcome",
    "verifier_latency_ms",
    "verifier_unavailable_units",
    "response_units",
    "verified_book_units",
    "repair_rounds",
    "repair_budget_exceeded",
    "turn_budget_exceeded",
    "all_required_supported",
    "adequacy_verdict",
    "failure_category",
    "failure_reason",
    "answers_request",
    "technically_grounded",
    "qualified",
    "answer_relevant",
    "selection_route",
    "semantic_selection_applied",
    "pack_order",
    "coverage_status",
    "evidence_discovered",
    "evidence_previewed",
    "evidence_read",
    "evidence_selected_in_pack",
    "evidence_used_in_answer",
    "evidence_quoted_to_user",
    "delivery_status",
    "product_outcome",
    "total_latency_ms",
    "reply_len",
)


def _ab_telemetry_snapshot(graph: Any, chat_id: int) -> dict[str, Any]:
    """Allow-listed production turn telemetry (never prompt/reply text)."""
    try:
        snapshot = graph.last_telemetry_for_thread(graph.thread_id(chat_id))
    except Exception:
        return {}
    if not isinstance(snapshot, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in _AB_TELEMETRY_KEYS:
        if key not in snapshot:
            continue
        value = snapshot[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
    return safe


def _ab_product_evidence(reply: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Independent product outcome and grounding evidence (no text persisted)."""
    product_outcome = "unknown"
    failure_reason = ""
    try:
        from aa.qualification.product_contract_live import delivered_product_accounting

        product_outcome, failure_reason = delivered_product_accounting(reply, snapshot)
    except Exception:
        product_outcome, failure_reason = "unknown", ""
    grounded: bool | None = None
    try:
        from aa.qualification.product_contract_live import (
            _is_grounded_substantive_reply,
        )

        grounded = bool(_is_grounded_substantive_reply(snapshot, reply))
    except Exception:
        grounded = None
    return {
        "product_outcome": str(product_outcome or "unknown")[:64],
        "failure_reason": str(failure_reason or "")[:64],
        "grounded_substantive": grounded,
        "reply_len": len(reply or ""),
        "reply_signature": _reply_signature(reply or ""),
    }


def _ab_missing_fields(arm: dict[str, Any]) -> list[str]:
    """Required observed fields; missing telemetry stays explicit, never inferred."""
    missing: list[str] = []
    if arm.get("delivered") is not True or arm.get("status") != "delivered":
        missing.append("delivery")
    if not isinstance(arm.get("effective_flag"), bool):
        missing.append("effective_flag")
    env_raw = str(arm.get("env_flag_raw", ""))
    expected = str(arm.get("expected_env_flag", ""))
    if expected not in ("0", "1"):
        missing.append("expected_env_flag")
    if env_raw not in ("0", "1"):
        missing.append("env_flag_raw")
    elif expected in ("0", "1") and env_raw != expected:
        missing.append("env_flag_effective")
    if (
        isinstance(arm.get("effective_flag"), bool)
        and expected in ("0", "1")
        and bool(arm["effective_flag"]) != (expected == "1")
    ):
        missing.append("effective_flag_mismatch")
    retrieval = arm.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, dict) else {}
    if not retrieval.get("metadata_present"):
        missing.append("retrieval.metadata")
    for key in ("retrieval_mode", "LLM_call_count", "coverage_status"):
        if key not in retrieval:
            missing.append(f"retrieval.{key}")
    attempts = arm.get("attempts")
    attempts = attempts if isinstance(attempts, dict) else {}
    if not attempts.get("present"):
        missing.append("attempts.ledger")
    if "model_http_attempts" not in attempts:
        missing.append("attempts.model_http_attempts")
    telemetry = arm.get("telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    if not telemetry:
        missing.append("telemetry.snapshot")
    for key in (
        "answer_outcome",
        "verifier_outcome",
        "verified_book_units",
        "adequacy_verdict",
    ):
        if key not in telemetry:
            missing.append(f"telemetry.{key}")
    product = arm.get("product")
    product = product if isinstance(product, dict) else {}
    if str(product.get("product_outcome", "unknown")) in ("", "unknown"):
        missing.append("product.product_outcome")
    if product.get("grounded_substantive") is None:
        missing.append("product.grounded_substantive")
    if not isinstance(arm.get("delivery_ms"), (int, float)):
        missing.append("delivery_ms")
    return missing


def _ab_should_stop_after_arm(arm: dict[str, Any]) -> tuple[bool, str]:
    """Stop before the ON arm on a confirmed provider block or non-delivery."""
    if arm.get("provider_429"):
        return True, "confirmed-429"
    if arm.get("active_backoff"):
        return True, "active-backoff"
    if arm.get("timed_out") or arm.get("status") == "timeout":
        return True, "timeout"
    if arm.get("status") == "bootstrap-failed":
        return True, "bootstrap-failed"
    if arm.get("status") != "delivered":
        return True, "arm-not-delivered"
    return False, ""


_AB_BLOCKING_SKIP_REASONS = frozenset(
    {
        "confirmed-429",
        "active-backoff",
        "timeout",
        "bootstrap-failed",
        "insufficient-job-budget",
    }
)


def classify_ab_outcome(
    arms: list[dict[str, Any]],
    *,
    second_arm_skipped_reason: str = "",
    deduplicated: bool = False,
) -> str:
    """Honest A/B status; no inferred speedup and no product PASS."""
    if deduplicated:
        return AB_STATUS_DEDUPLICATED
    if second_arm_skipped_reason in _AB_BLOCKING_SKIP_REASONS:
        return AB_STATUS_BLOCKED
    if not arms:
        return AB_STATUS_BLOCKED if second_arm_skipped_reason else AB_STATUS_INCONCLUSIVE
    if any(
        arm.get("provider_429")
        or arm.get("active_backoff")
        or arm.get("timed_out")
        or arm.get("status") in ("timeout", "bootstrap-failed")
        for arm in arms
    ):
        return AB_STATUS_BLOCKED
    if len(arms) < 2:
        return AB_STATUS_INCONCLUSIVE
    if any(arm.get("status") != "delivered" for arm in arms):
        return AB_STATUS_INCONCLUSIVE
    if any(_ab_missing_fields(arm) for arm in arms):
        return AB_STATUS_INCOMPLETE
    return AB_STATUS_COMPLETE


def _emit_ab_arm_result(
    sink: EventSink,
    *,
    arm: str,
    probe_label: str,
    state: dict[str, Any],
    app: Any,
    graph: Any,
    scenario: dict[str, Any],
    budget: dict[str, float],
) -> None:
    """Emit one privacy-safe arm result (the parent's only A/B evidence)."""
    attempts: Any = ()
    request_audit: Any = ()
    try:
        client = app.opencode_runtime.client
        attempts = getattr(client, "attempt_audit", ())
        request_audit = getattr(client, "request_latency_audit", ())
    except Exception:
        attempts = ()
        request_audit = ()
    telemetry = state.get("telemetry")
    if not isinstance(telemetry, dict) or not telemetry:
        telemetry = _ab_telemetry_snapshot(graph, int(scenario.get("chat_id", SCENARIO_2_CHAT_ID)))
    retrieval = _ab_retrieval_evidence(state.get("capture"))
    attempt_summary = _ab_attempt_summary(attempts, request_audit)
    product = state.get("product")
    if not isinstance(product, dict):
        product = {
            "product_outcome": "unknown",
            "failure_reason": "",
            "grounded_substantive": None,
            "reply_len": 0,
            "reply_signature": "",
        }
    provider_429 = bool(retrieval.get("provider_429") or attempt_summary.get("provider_429"))
    active_backoff = bool(attempt_summary.get("active_backoff") or provider_429)
    payload: dict[str, Any] = {
        "arm": arm,
        "probe": probe_label,
        "mode": MODE_ONE_SHOT_AB,
        "status": str(state.get("status", "") or "failed")[:48],
        "delivered": bool(state.get("delivered", False)),
        "timed_out": bool(state.get("timed_out", False)),
        "typed_failure": str(state.get("typed_failure", "") or "")[:64],
        "env_flag_raw": str(state.get("env_flag_raw", "") or "")[:8],
        "expected_env_flag": expected_ab_env_flag(arm),
        "effective_flag": state.get("effective_flag"),
        "scenario": dict(scenario),
        "budget": dict(budget),
        "preflight_elapsed_ms": state.get("preflight_elapsed_ms"),
        "live_elapsed_ms": state.get("live_elapsed_ms"),
        "delivery_ms": state.get("delivery_ms"),
        "retrieval": retrieval,
        "attempts": attempt_summary,
        "telemetry": telemetry,
        "product": product,
        "provider_429": provider_429,
        "active_backoff": active_backoff,
    }
    payload["missing_fields"] = _ab_missing_fields(payload)
    sink.emit("ab-arm-result", **payload)
    state["emitted"] = True


async def run_probe_b(argv: dict[str, Any], sink: EventSink, counters: dict[str, Any]) -> int:
    """Probe B child: exact production scenario #2 (unchanged prompt).

    kodmial/aa#424: the standalone scenario2 observation has NO wall-clock
    cutoff. The kodmial/aa#553 A/B arm additionally honors the existing
    bounded canary budget (300 s live / 600 s preflight / cleanup reserve)
    because the owner-approved A/B is explicitly validation-only.
    """
    from dataclasses import replace

    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.semantic_work_plan import set_stage_observer
    from aa.telegram.transport import PollingTelegramTransport

    probe_label = str(argv.get("probe") or PROBE_SCENARIO2)
    ab_mode = bool(argv.get("ab_mode", False))
    ab_arm = str(argv.get("ab_arm", "") or "")
    ab_budget = _ab_budget(
        float(argv.get("ab_live_budget_s", DEFAULT_AB_LIVE_BUDGET_S)),
        float(argv.get("ab_preflight_budget_s", DEFAULT_AB_PREFLIGHT_BUDGET_S)),
        float(argv.get("ab_cleanup_reserve_s", DEFAULT_AB_CLEANUP_RESERVE_S)),
    )
    if ab_mode:
        scenario_family, scenario_chat_id, scenario_prompt, _scenario_source = (
            _ab_canonical_scenario()
        )
    else:
        scenario_family = SCENARIO_2_FAMILY
        scenario_chat_id = SCENARIO_2_CHAT_ID
        scenario_prompt = SCENARIO_2_PROMPT
    scenario_identity = ab_scenario_identity() if ab_mode else {}
    base_url = os.environ.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096")
    out_dir = Path(argv["out_dir"])
    counters["snapshot_index"] = 0
    anchor = threading.Event()
    bootstrap_timeout = float(
        argv.get("probe_b_bootstrap_timeout_s", DEFAULT_PROBE_B_BOOTSTRAP_TIMEOUT_S)
    )
    if ab_mode:
        bootstrap_timeout = min(bootstrap_timeout, ab_budget["preflight_budget_s"])
    watchdog = ChildWatchdog(
        probe=probe_label,
        cadence_s=float(argv.get("snapshot_cadence_s", DEFAULT_SNAPSHOT_CADENCE_S)),
        max_snapshots=int(argv.get("max_snapshots", DEFAULT_MAX_SNAPSHOTS)),
        anchor=anchor,
        anchor_wait_timeout_s=max(120.0, bootstrap_timeout + 30.0),
        sink=sink,
        checkpoint_dir=out_dir,
        opencode_base_url=base_url,
        counters=counters,
        internal_probe=lambda: _opencode_internal_observation(base_url, ""),
    )
    watchdog.main_loop = asyncio.get_running_loop()
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
    sink.emit(
        "probe-start",
        probe=probe_label,
        family=scenario_family,
        arm=ab_arm if ab_mode else "",
    )
    counters["last_real_event"] = "probe-start"
    counters["last_real_monotonic"] = time.monotonic()
    exit_code = EXIT_OK
    continuity_state: dict[str, Any] = {"index": 0, "next_at": 0.0}
    graph: Any = None
    ab_capture: dict[str, Any] | None = None
    ab_original: Any = None
    ab_state: dict[str, Any] = {
        "emitted": False,
        "status": "running",
        "delivered": False,
        "timed_out": False,
        "typed_failure": "",
        "env_flag_raw": os.environ.get(ONE_SHOT_FASTPATH_ENV, ""),
        "effective_flag": None,
        "preflight_elapsed_ms": None,
        "live_elapsed_ms": None,
        "delivery_ms": None,
        "capture": None,
        "telemetry": {},
        "product": {},
    }
    if ab_mode:
        try:
            ab_capture, ab_original = _install_ab_retrieval_probe()
            ab_state["capture"] = ab_capture
            from aa.conversation.retrieval_node import one_shot_fast_path_enabled

            ab_state["effective_flag"] = bool(one_shot_fast_path_enabled())
        except Exception as exc:
            ab_state["typed_failure"] = "instrumentation:" + type(exc).__name__[:48]
    try:
        sink.emit("bootstrap-start", probe=probe_label)
        counters["last_real_event"] = "bootstrap-start"
        counters["last_real_monotonic"] = time.monotonic()
        started = time.perf_counter()
        try:
            await asyncio.wait_for(app.start(), timeout=bootstrap_timeout)
        except Exception as exc:
            ab_state["status"] = "bootstrap-failed"
            ab_state["typed_failure"] = "bootstrap:" + type(exc).__name__[:48]
            ab_state["preflight_elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            sink.emit(
                "bootstrap-end",
                ok=False,
                elapsed_ms=ab_state["preflight_elapsed_ms"],
                fail_category=type(exc).__name__[:48],
            )
            sink.emit("probe-end", probe=probe_label, status="bootstrap-failed")
            return EXIT_BOOTSTRAP_FAILED
        ab_state["preflight_elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        counters["last_real_event"] = "bootstrap-end"
        counters["last_real_monotonic"] = time.monotonic()
        graph = app.graph_runtime
        sink.emit(
            "bootstrap-end",
            ok=True,
            elapsed_ms=ab_state["preflight_elapsed_ms"],
            opencode_ready=bool(app.opencode_runtime.running and app.opencode_runtime.ready),
            graph_running=bool(graph is not None and graph.running),
        )
        watchdog.audit_provider = lambda: list(app.opencode_runtime.client.request_latency_audit)
        set_stage_observer(observer)
        before = len(api.sent_texts)
        sink.emit(
            "scenario-start",
            family=scenario_family,
            scenario_index=2,
            chat_id=scenario_chat_id,
            arm=ab_arm if ab_mode else "",
        )
        counters["last_real_event"] = "scenario-start"
        counters["last_real_monotonic"] = time.monotonic()
        raw = {
            "update_id": 930103,
            "message": {
                "message_id": 2,
                "date": 1,
                "chat": {"id": scenario_chat_id, "type": "private"},
                "text": scenario_prompt,
            },
        }
        api.pending_updates.append(raw)
        scenario_started = time.monotonic()
        live_deadline = (
            scenario_started + float(ab_budget["live_model_budget_s"]) if ab_mode else None
        )
        anchor.set()
        # kodmial/aa#424: no cutoff in standalone mode. The A/B arm is the
        # explicit exception required by #553: its existing 300 s live canary
        # budget bounds the arm, with the cleanup reserve kept inside it.
        while len(api.sent_texts) <= before:
            counters["typing_sends"] = api.chat_actions
            if live_deadline is not None and time.monotonic() >= live_deadline:
                ab_state["timed_out"] = True
                ab_state["status"] = "timeout"
                ab_state["typed_failure"] = "timeout"
                ab_state["live_elapsed_ms"] = round(
                    (time.monotonic() - scenario_started) * 1000.0, 1
                )
                sink.emit(
                    "probe-end",
                    probe=probe_label,
                    status="timeout",
                    fail_category="live-budget-exhausted",
                )
                exit_code = EXIT_INCONCLUSIVE
                return exit_code
            if counters.get("server_nonresponsive_sustained"):
                ab_state["status"] = CLASS_SERVER_ERROR_PROVEN
                ab_state["typed_failure"] = "server-nonresponsive-sustained"
                ab_state["live_elapsed_ms"] = round(
                    (time.monotonic() - scenario_started) * 1000.0, 1
                )
                sink.emit(
                    "probe-end",
                    probe=probe_label,
                    status=CLASS_SERVER_ERROR_PROVEN,
                    fail_category="server-nonresponsive-sustained",
                )
                exit_code = EXIT_SERVER_ERROR_PROVEN
                return exit_code
            _emit_observation_continuity(sink, counters, scenario_started, continuity_state)
            await asyncio.sleep(0.05)
        elapsed_ms = round((time.monotonic() - scenario_started) * 1000.0, 1)
        counters["delivered"] = True
        counters["last_real_event"] = "delivery"
        counters["last_real_monotonic"] = time.monotonic()
        reply = api.sent_texts[-1]
        ab_state["status"] = "delivered"
        ab_state["delivered"] = True
        ab_state["delivery_ms"] = elapsed_ms
        ab_state["live_elapsed_ms"] = elapsed_ms
        sink.emit(
            "delivery",
            elapsed_ms=elapsed_ms,
            reply_len=len(reply),
            reply_signature=_reply_signature(reply),
            typing_sends=api.chat_actions,
        )
        ab_state["telemetry"] = _ab_telemetry_snapshot(graph, scenario_chat_id)
        ab_state["product"] = _ab_product_evidence(reply, ab_state["telemetry"])
        stage_values = _collect_stage_telemetry(graph, scenario_chat_id)
        sink.emit(
            "scenario-end",
            family=scenario_family,
            stage_values=len(stage_values),
            arm=ab_arm if ab_mode else "",
        )

        sink.emit("probe-end", probe=probe_label, status="delivered")
        exit_code = EXIT_OK
        return exit_code
    finally:
        set_stage_observer(None)
        if ab_mode:
            if ab_state.get("status") in ("", "running"):
                ab_state["status"] = "failed"
                if not ab_state.get("typed_failure"):
                    ab_state["typed_failure"] = "unexpected"
            if not ab_state.get("emitted"):
                try:
                    _emit_ab_arm_result(
                        sink,
                        arm=ab_arm,
                        probe_label=probe_label,
                        state=ab_state,
                        app=app,
                        graph=graph,
                        scenario=scenario_identity,
                        budget=ab_budget,
                    )
                except Exception as exc:
                    try:
                        sink.emit(
                            "ab-result-emit-failed",
                            probe=probe_label,
                            arm=ab_arm,
                            category=type(exc).__name__[:48],
                        )
                    except Exception:
                        pass
            if ab_original is not None:
                _restore_ab_retrieval_probe(ab_original)
        stopped = await _bounded_async_stop(app, bootstrap_timeout)
        anchor.set()
        sink.emit("child-exit", probe=probe_label, exit_code=exit_code, cleanup_done=stopped)
        watchdog.stop()
        if not stopped:
            _forced_bounded_exit(sink, probe_label, exit_code)


def _collect_stage_telemetry(graph: Any, chat_id: int = SCENARIO_2_CHAT_ID) -> list[dict[str, Any]]:
    """Sanitized per-stage turn telemetry (values only, no stage text)."""
    values: list[dict[str, Any]] = []
    if graph is None:
        return values
    try:
        telemetry = graph.last_telemetry_for_thread(graph.thread_id(chat_id))
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
    registered = _install_stack_handler(
        out_dir / f"stacks-{probe}.txt",
        warn=lambda message: sink.emit("stack-handler-failed", detail=str(message)[:64]),
    )
    # The parent may only SIGUSR1 the child AFTER this sentinel exists: an
    # unregistered child would die by default SIGUSR1 disposition.
    if registered:
        try:
            _stack_handler_ready_path(out_dir, probe).write_text(now_iso() + "\n", encoding="utf-8")
        except OSError:
            pass
    counters: dict[str, Any] = {"snapshot_index": 0}
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if probe == PROBE_DIRECT_MODEL:
            return int(loop.run_until_complete(run_probe_a(argv, sink, counters)))
        if probe in SCENARIO2_PROBES:
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

    def __init__(
        self,
        owner: str,
        repo: str,
        issue: int,
        token: str,
        run_id: str = "",
        *,
        marker: str = "",
    ):
        self.owner = owner or DEFAULT_OWNER
        self.repo = repo or DEFAULT_REPO
        self.issue = int(issue or DEFAULT_ISSUE)
        self.token = (token or "").strip()
        self.run_id = run_id or os.environ.get("GITHUB_RUN_ID", "") or "local"
        self.marker = marker or comment_marker(self.run_id)
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

    def list_comments(self) -> list[dict[str, Any]] | None:
        """Bounded first page of issue comments (fail-soft)."""
        ok, comments = self._request(
            "GET",
            f"/repos/{self.owner}/{self.repo}/issues/{self.issue}/comments?per_page=100",
        )
        if not ok or not isinstance(comments, list):
            return None
        return [comment for comment in comments[:100] if isinstance(comment, dict)]

    def ensure_comment(self) -> int | None:
        """Find this run's existing diagnostic comment (bounded, idempotent)."""
        if self._comment_id is not None:
            return self._comment_id
        comments = self.list_comments()
        if comments is None:
            return None
        for comment in comments:
            body = comment.get("body")
            if isinstance(body, str) and self.marker in body:
                comment_id = comment.get("id")
                if isinstance(comment_id, int):
                    self._comment_id = comment_id
                    return comment_id
        return None

    def publish(self, title: str, body: str, *, force: bool = False) -> bool:
        """Create once, then update the same comment (throttled, fail-soft)."""
        safe_title = title.replace("|", "/").replace("\n", " ")[:64]
        stamped = f"{self.marker}\n**{safe_title}**\n\n{body}\n"
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
# Parent mode: campaign supervision, cadence comments and the final report
# ---------------------------------------------------------------------------


@dataclass
class ProbeEvidence:
    """Privacy-safe evidence state assembled by the supervisor thread."""

    probe: str
    status: str = "not-run"
    exit_code: int | None = None
    killed: bool = False
    # True only when the child was still alive at the campaign physical
    # allowance and the supervisor itself ended it (bounded tidy shutdown,
    # not abnormal): the outcome is unresolved-at-campaign-end.
    terminated_at_campaign_end: bool = False
    # True when the child itself requested a documented bounded process-group
    # end (cleanup could not be released); never a harness fault.
    forced_exit: bool = False
    # Sustained (proven) local-server non-responsiveness observed by the
    # child watchdog; local infrastructure evidence, never provider evidence.
    server_nonresponsive_sustained: bool = False
    last_stage: str = ""
    last_stage_at_s: float | None = None
    seconds_since_last_stage: float | None = None
    # Most recent IN-FLIGHT stage reported by a child ``stage-start`` event
    # (e.g. ``opencode`` while its HTTP reply is pending) — separate from
    # the snapshot/checkpoint stage, which may be stale.
    active_stage: str = ""
    active_stage_ts: str = ""
    # Last stage the child closed with a ``stage-end`` event.
    last_completed_stage: str = ""
    last_completed_stage_elapsed_ms: float | None = None
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
    # kodmial/aa#553: the child's single privacy-safe A/B arm result. Empty
    # for the standalone scenario2 probe.
    ab_result: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_signal(self) -> int | None:
        """Positive signal number when the child died by signal (POSIX)."""
        code = self.exit_code
        if code is not None and code < 0 and -int(code) <= 64:
            return -int(code)
        return None

    @property
    def last_boundary_stage(self) -> str:
        """The most recent observed boundary stage (never a root cause)."""
        return (self.active_stage or self.last_stage or "").strip()


@dataclass
class ProbePlan:
    """One probe's supervision parameters (NO per-answer cutoff exists)."""

    probe: str
    snapshot_cadence_s: float
    max_snapshots: int
    connect_timeout_s: float
    # kodmial/aa#553: explicit per-child environment (the one-shot flag is
    # set only inside the A/B child, never in the parent or production).
    arm_env: dict[str, str] = field(default_factory=dict)
    ab_mode: bool = False
    ab_arm: str = ""
    ab_live_budget_s: float = DEFAULT_AB_LIVE_BUDGET_S
    ab_preflight_budget_s: float = DEFAULT_AB_PREFLIGHT_BUDGET_S
    ab_cleanup_reserve_s: float = DEFAULT_AB_CLEANUP_RESERVE_S


def effective_campaign_allowance_s(
    *,
    campaign_allowance_s: float,
    finalisation_reserve_s: float,
    job_timeout_minutes: float = 0.0,
    job_started_epoch: float = 0.0,
    now_epoch: float | None = None,
) -> float:
    """The ONE finite campaign ceiling (no per-answer wall clock anywhere).

    Reconciled with the remaining Actions job budget when a job-start
    marker is provided so the finalisation reserve and artifact upload
    always complete inside the job. If that budget is already inside the
    finalisation reserve, the allowance keeps a minimum viable observation
    window instead of collapsing to a token one second.
    """
    configured = float(campaign_allowance_s)
    if not math.isfinite(configured):
        configured = MIN_CAMPAIGN_ALLOWANCE_S
    configured = max(MIN_CAMPAIGN_ALLOWANCE_S, configured)
    timeout_minutes = float(job_timeout_minutes or 0.0)
    started_epoch = float(job_started_epoch or 0.0)
    if timeout_minutes <= 0.0 or started_epoch <= 0.0:
        return configured
    current = time.time() if now_epoch is None else now_epoch
    remaining = started_epoch + timeout_minutes * 60.0 - float(finalisation_reserve_s) - current
    if not math.isfinite(remaining):
        return configured
    if remaining <= 0.0:
        return MIN_CAMPAIGN_ALLOWANCE_S
    return max(MIN_CAMPAIGN_ALLOWANCE_S, min(configured, remaining))


def _spawn_child(*, plan: ProbePlan, out_dir: Path) -> subprocess.Popen:
    """Start one probe child in its own process group (independent)."""
    child_argv = json.dumps(
        {
            "probe": plan.probe,
            "out": str(out_dir),
            "snapshot_cadence_s": plan.snapshot_cadence_s,
            "max_snapshots": plan.max_snapshots,
            "connect_timeout_s": plan.connect_timeout_s,
            "ab_mode": plan.ab_mode,
            "ab_arm": plan.ab_arm,
            "ab_live_budget_s": plan.ab_live_budget_s,
            "ab_preflight_budget_s": plan.ab_preflight_budget_s,
            "ab_cleanup_reserve_s": plan.ab_cleanup_reserve_s,
        }
    )
    env = os.environ.copy()
    env["SCENARIO2_DIAG_ARGS"] = child_argv
    env["PYTHONUNBUFFERED"] = "1"
    # #553: each arm's flag is explicit and isolated inside the child only.
    env.update({str(key): str(value) for key, value in plan.arm_env.items()})
    # The child inherits stdout (safe JSONL) and writes its own artifact
    # file. Never use a pipe here: a slow reader could fill the pipe buffer
    # and block the child.
    return subprocess.Popen(  # noqa: S603 (fixed argv, no shell)
        [sys.executable, str(Path(__file__).resolve()), "--child"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=None,
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
        "snapshot_cadence_s": float(decoded.get("snapshot_cadence_s", DEFAULT_SNAPSHOT_CADENCE_S)),
        "max_snapshots": int(decoded.get("max_snapshots", DEFAULT_MAX_SNAPSHOTS)),
        "connect_timeout_s": float(
            decoded.get("connect_timeout_s", DEFAULT_PROBE_A_CONNECT_TIMEOUT_S)
        ),
        "probe_a_port": str(os.environ.get("SCENARIO2_DIAG_PROBE_A_PORT", DEFAULT_PROBE_A_PORT)),
        "probe_a_bootstrap_timeout_s": float(
            os.environ.get("SCENARIO2_DIAG_PROBE_A_BOOTSTRAP_TIMEOUT_S", "")
            or DEFAULT_PROBE_A_BOOTSTRAP_TIMEOUT_S
        ),
        "probe_b_bootstrap_timeout_s": float(
            os.environ.get("SCENARIO2_DIAG_PROBE_B_BOOTSTRAP_TIMEOUT_S", "")
            or DEFAULT_PROBE_B_BOOTSTRAP_TIMEOUT_S
        ),
        "ab_mode": bool(decoded.get("ab_mode", False)),
        "ab_arm": str(decoded.get("ab_arm", "")),
        "ab_live_budget_s": float(decoded.get("ab_live_budget_s", DEFAULT_AB_LIVE_BUDGET_S)),
        "ab_preflight_budget_s": float(
            decoded.get("ab_preflight_budget_s", DEFAULT_AB_PREFLIGHT_BUDGET_S)
        ),
        "ab_cleanup_reserve_s": float(
            decoded.get("ab_cleanup_reserve_s", DEFAULT_AB_CLEANUP_RESERVE_S)
        ),
    }


def _child_events_path(out_dir: Path, probe: str) -> Path:
    """Artifact file the child writes its safe JSONL events to."""
    return out_dir / f"probe-{probe}-events.jsonl"


def _consume_child_line(line: str, evidence: ProbeEvidence, sink: EventSink) -> bool:
    """Parse one child JSONL line; return ``False`` on a terminal event."""
    line = line.strip()
    if not line:
        return True
    sink.passthrough(line)
    try:
        payload = json.loads(line)
    except ValueError:
        evidence.raw_lines += 1
        return evidence.raw_lines <= 200  # bounded raw non-JSON capture
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
    elif kind == "observation-continues":
        # Evidence checkpoint: observation continued past a legacy window.
        if len(evidence.timeline) < 48:
            evidence.timeline.append(
                {
                    "at_s": payload.get("threshold_s"),
                    "stage": "observation-continues",
                    "since_stage_s": payload.get("seconds_since_last_real"),
                    "typing_sends": None,
                    "delivered": bool(payload.get("delivered", False)),
                }
            )
    elif kind == "server-health":
        evidence.server_ready_ms = payload.get("elapsed_ms")
        if payload.get("ok") is False:
            evidence.fail_category = "server-health-failed"
    elif kind == "stage-start":
        # In-flight stage marker from the child graph observer: the
        # freshest boundary evidence. NEVER a proven root cause by itself.
        stage = str(payload.get("stage", ""))[:64]
        if stage:
            evidence.active_stage = stage
            evidence.active_stage_ts = str(payload.get("ts", ""))[:20]
    elif kind == "stage-end":
        stage = str(payload.get("stage", ""))[:64]
        if stage:
            evidence.last_completed_stage = stage
            elapsed = payload.get("elapsed_ms")
            if isinstance(elapsed, (int, float)):
                evidence.last_completed_stage_elapsed_ms = float(elapsed)
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
        evidence.fail_category = "bootstrap:" + str(payload.get("fail_category", "unknown"))[:48]
        evidence.status = "bootstrap-failed"
    elif kind == "server-nonresponsive-sustained":
        evidence.server_nonresponsive_sustained = True
        if not evidence.fail_category:
            evidence.fail_category = "server-nonresponsive-sustained"
    elif kind == "child-forced-exit":
        evidence.forced_exit = True
    elif kind == "ab-arm-result":
        # #553: one privacy-safe arm result; the parent never infers fields.
        evidence.ab_result = {
            key: value for key, value in payload.items() if key not in ("ts", "mono")
        }
        status = str(payload.get("status", "") or "")
        if status:
            evidence.status = status[:48]
        if payload.get("delivered") is True:
            evidence.delivered = True
        if payload.get("timed_out") is True:
            evidence.terminated_at_campaign_end = True
    elif kind == "ab-result-emit-failed":
        evidence.fail_category = (
            "ab-result-emit-failed:" + str(payload.get("category", "unknown"))[:32]
        )
    elif kind == "probe-end":
        evidence.status = str(payload.get("status", ""))[:48]
    elif kind == "child-exit":
        return False
    return True


def _tail_events_file(
    path: Path,
    evidence: ProbeEvidence,
    sink: EventSink,
    proc: subprocess.Popen | None,
) -> None:
    """Tail the child artifact file without ever writing into a child pipe.

    Reading a regular file never applies backpressure to the child, so even
    a burst of JSONL events cannot block it.
    """
    while not path.exists():
        if proc is not None and proc.poll() is not None:
            return
        time.sleep(0.02)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            while True:
                line = stream.readline()
                if line:
                    if not _consume_child_line(line, evidence, sink):
                        return
                    continue
                if proc is None or proc.poll() is not None:
                    return
                time.sleep(0.02)
    except OSError:
        return


def _reader_loop(
    source: Any,
    evidence: ProbeEvidence,
    sink: EventSink,
    proc: subprocess.Popen | None = None,
) -> None:
    """Consume child JSONL events into the Actions log + probe state."""
    if isinstance(source, (str, os.PathLike)):
        _tail_events_file(Path(source), evidence, sink, proc)
        return
    try:
        for line in source:
            if not _consume_child_line(str(line), evidence, sink):
                break
    finally:
        try:
            source.close()
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


def _send_sigusr1(proc: subprocess.Popen, *, ready_path: Path | None = None) -> bool:
    """Request a whole-process faulthandler dump inside the child.

    Never signals a child that has not proven (via its ready sentinel) that
    a SIGUSR1 handler is registered: an unregistered child dies by the
    default SIGUSR1 termination disposition.
    """
    if ready_path is not None and not ready_path.exists():
        return False
    try:
        proc.send_signal(signal.SIGUSR1)
        return True
    except (ValueError, OSError):
        return False


def _final_supervisor_snapshot(
    out_dir: Path,
    proc: subprocess.Popen,
    evidence: ProbeEvidence,
    *,
    kind: str,
    ready_path: Path | None = None,
) -> dict[str, Any]:
    """Terminal snapshot collected by the supervisor (independent of child)."""
    _send_sigusr1(proc, ready_path=ready_path)
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
    if evidence.active_stage:
        rows.append(f"- active_stage: `{evidence.active_stage}` (in-flight; boundary only)")
        if evidence.active_stage_ts:
            rows.append(f"- active_stage_started_at: `{evidence.active_stage_ts}`")
    if evidence.last_completed_stage:
        rows.append(f"- last_completed_stage: `{evidence.last_completed_stage}`")
    if evidence.last_completed_stage_elapsed_ms is not None:
        rows.append(f"- last_completed_stage_ms: {evidence.last_completed_stage_elapsed_ms}")
    if evidence.fail_category:
        rows.append(f"- fail_category: `{evidence.fail_category}`")
    if evidence.server_nonresponsive_sustained:
        rows.append("- server_nonresponsive_sustained: true (proven local-infra)")
    if evidence.last_real_event:
        rows.append(f"- last_real_event: `{evidence.last_real_event}`")
    rows.append(f"- snapshots: {evidence.snapshots}")
    rows.append(f"- killed: {str(evidence.killed).lower()}")
    signal_number = evidence.exit_signal
    if signal_number is not None and not evidence.forced_exit:
        rows.append(f"- exited_by_signal: SIG{signal_number}")
    elif evidence.terminated_at_campaign_end:
        rows.append("- terminated_at_campaign_end: true (campaign allowance; evidence kept)")
    return rows


def _publish_probe_update(publisher: IssueProgressPublisher, evidence: ProbeEvidence) -> None:
    body = "\n".join(_probe_comment_lines(evidence))
    publisher.publish(f"Scenario #2 diagnostic: {evidence.probe}", body)


def _parent_observation_continuity(
    sink: EventSink,
    evidence: ProbeEvidence,
    started: float,
    state: dict[str, Any],
) -> None:
    """Parent-side observability: observation continues past old windows."""
    elapsed = time.monotonic() - started
    index = int(state.get("index", 0))
    thresholds = OBSERVATION_CONTINUITY_THRESHOLD_S
    if index < len(thresholds):
        if elapsed < thresholds[index]:
            return
        threshold = thresholds[index]
        state["index"] = index + 1
        state["next_at"] = elapsed + OBSERVATION_CONTINUITY_REPEAT_S
    else:
        if elapsed < float(state.get("next_at", 0.0)):
            return
        threshold = elapsed
        state["next_at"] = elapsed + OBSERVATION_CONTINUITY_REPEAT_S
    sink.emit(
        "observation-continues",
        probe=evidence.probe,
        elapsed_s=round(elapsed, 1),
        threshold_s=round(float(threshold), 1),
        last_real_event=str(evidence.last_real_event)[:48],
    )


def _run_one_probe(
    plan: ProbePlan,
    *,
    out_dir: Path,
    sink: EventSink,
    publisher: IssueProgressPublisher | None,
    campaign_deadline_s: float | None,
) -> ProbeEvidence:
    """Supervise one probe child end-to-end (isolated, campaign-bounded).

    kodmial/aa#424: the supervisor NEVER interrupts a live child with a
    wall-clock answer cutoff. The only bounded physical enforcement is the
    shared campaign deadline (tidy shutdown + evidence upload); a
    genuinely progressing turn at that boundary is reported
    ``unresolved-at-campaign-end`` — evidence plus a resumable allowance,
    never a component verdict.
    """
    evidence = ProbeEvidence(probe=plan.probe)
    evidence.status = "running"
    sink.emit(
        "supervisor-probe-start",
        probe=plan.probe,
        campaign_time_left_s=(
            None
            if campaign_deadline_s is None
            else round(campaign_deadline_s - time.monotonic(), 1)
        ),
        snapshot_cadence_s=plan.snapshot_cadence_s,
        snapshot_max=plan.max_snapshots,
    )
    proc = _spawn_child(plan=plan, out_dir=out_dir)
    reader = threading.Thread(
        target=_reader_loop,
        args=(_child_events_path(out_dir, plan.probe), evidence, sink, proc),
        daemon=True,
    )
    reader.start()
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    sink.emit("supervisor-child-spawn", probe=plan.probe, has_group=pgid is not None)
    started = time.monotonic()
    # The supervisor only SIGUSR1s the child after the child wrote its
    # stack-handler-ready sentinel (faulthandler registered, chain=False);
    # an unregistered child would die by default SIGUSR1 disposition.
    stack_ready_path = _stack_handler_ready_path(out_dir, plan.probe)
    last_stack_request = started
    stack_dumps = 0
    last_publish = started
    continuity_state: dict[str, Any] = {"index": 0, "next_at": 0.0}
    while proc.poll() is None:
        now = time.monotonic()
        if not evidence.delivered:
            _parent_observation_continuity(sink, evidence, started, continuity_state)
        if campaign_deadline_s is not None and now >= campaign_deadline_s:
            sink.emit(
                "campaign-end",
                probe=plan.probe,
                elapsed_s=round(now - started, 1),
                delivered=evidence.delivered,
                last_boundary_stage=evidence.last_boundary_stage[:64],
            )
            _final_supervisor_snapshot(
                out_dir,
                proc,
                evidence,
                kind=f"supervisor-final-snapshot-{plan.probe}",
                ready_path=stack_ready_path,
            )
            evidence.killed = _kill_group(proc, DEFAULT_TERMINATE_GRACE_S)
            evidence.terminated_at_campaign_end = True
            break
        if stack_dumps < MAX_STACK_DUMPS and now - last_stack_request >= STACK_DUMP_EVERY_S:
            _send_sigusr1(proc, ready_path=stack_ready_path)
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
        if evidence.delivered:
            evidence.status = "delivered"
        elif evidence.terminated_at_campaign_end:
            evidence.status = CLASS_UNRESOLVED
        else:
            evidence.status = f"child-exit-{proc.returncode}"
    if publisher is not None:
        _publish_probe_update(publisher, evidence)
    sink.emit(
        "supervisor-probe-end",
        probe=plan.probe,
        status=evidence.status,
        exit_code=None if evidence.exit_code is None else int(evidence.exit_code),
        exited_by_signal=evidence.exit_signal,
        terminated_at_campaign_end=evidence.terminated_at_campaign_end,
        killed=evidence.killed,
        delivered=evidence.delivered,
        snapshots=evidence.snapshots,
        active_stage=evidence.active_stage,
        last_completed_stage=evidence.last_completed_stage,
    )
    return evidence


# ---------------------------------------------------------------------------
# Classification and the compact root-cause report
# ---------------------------------------------------------------------------


INFRASTRUCTURE_FAILURE = CLASS_HARNESS_ERROR


def _abnormal_child_exit(evidence: ProbeEvidence) -> bool:
    """True when a child died by an unexpected signal or crash (harness fault).

    A campaign-allowance shutdown (``SIGTERM``/``SIGKILL`` of the whole
    group at the finite physical boundary) is NOT abnormal: it is the
    documented evidence-keeping INCONCLUSIVE path. A documented forced
    bounded child exit (cleanup timeout, event already flushed) is also
    not abnormal. A negative non-forced exit code (e.g. ``-10`` == a
    chained SIGUSR1 disposition) or an unexpected crash exit means the
    diagnostic harness truncated its own probe — stage markers from that
    child are NOT proven cause evidence.
    """
    if evidence.terminated_at_campaign_end or evidence.status in ("not-run", ""):
        return False
    code = evidence.exit_code
    if code is None:
        return False
    if code < 0:
        return not evidence.forced_exit
    return code in (EXIT_CHILD_UNEXPECTED, EXIT_REPORT_FAILED)


def classify_cause(probe_a: ProbeEvidence, probe_b: ProbeEvidence) -> str:
    """Evidence-driven classification — truthful verdict taxonomy (#424).

    Never blames model/provider from a stage name: an in-flight
    ``opencode`` stage with no observed event is an UNOBSERVED request —
    BOUNDARY LOCALIZED / ROOT UNKNOWN, never ``model-provider``.
    """
    # HARNESS ERROR: the diagnostic harness truncated its own probes.
    if _abnormal_child_exit(probe_a) or _abnormal_child_exit(probe_b):
        return CLASS_HARNESS_ERROR
    # DELIVERED: the scenario turn itself delivered (probe evidence kept).
    if probe_b.delivered or probe_b.status == "delivered":
        return CLASS_DELIVERED
    # PROVEN provider error: a real error payload was observed by the
    # direct probe while the turn was undelivered.
    if probe_a.status == CLASS_PROVIDER_ERROR_PROVEN:
        return CLASS_PROVIDER_ERROR_PROVEN
    # PROVEN local server error (spawn/health/transport evidence only).
    if probe_a.status == CLASS_SERVER_ERROR_PROVEN or probe_b.status == (CLASS_SERVER_ERROR_PROVEN):
        return CLASS_SERVER_ERROR_PROVEN
    # B-probe bootstrap failure: the production path failed before the
    # scenario could be observed; the root is unproven.
    if probe_b.status == "bootstrap-failed":
        return CLASS_BOUNDARY_LOCALIZED
    # UNRESOLVED AT CAMPAIGN END: the physical allowance was reached with
    # no delivered answer and no proven failure — live-but-uncertain.
    if (
        probe_a.terminated_at_campaign_end
        or probe_b.terminated_at_campaign_end
        or probe_a.status == CLASS_UNRESOLVED
        or probe_b.status == CLASS_UNRESOLVED
    ):
        return CLASS_UNRESOLVED
    # Boundary localization only: name the observed boundary, never a
    # component root cause.
    if not probe_b.last_boundary_stage:
        return CLASS_STAGE_UNOBSERVED
    return CLASS_BOUNDARY_LOCALIZED


def overall_exit_code(probe_a: ProbeEvidence, probe_b: ProbeEvidence) -> int:
    """Truthful supervisor exit code derived from the classification."""
    classification = classify_cause(probe_a, probe_b)
    return _EXIT_BY_CLASSIFICATION.get(classification, EXIT_INCONCLUSIVE)


_NEXT_FIX: dict[str, str] = {
    CLASS_SERVER_ERROR_PROVEN: (
        "Inspect the supervised `opencode serve` spawn/health/transport path "
        "(LocalOpenCodeRuntime) on this runner: verify the pinned binary "
        "version and health timing. Proven local-server failure only; no "
        "product timeout changes."
    ),
    CLASS_PROVIDER_ERROR_PROVEN: (
        "The direct model call returned a real error payload; investigate "
        "the provider/model boundary for the approved AA bot model "
        "(quota/429 handling and served-model identity) before touching "
        "retrieval. No product timeout changes."
    ),
    CLASS_HARNESS_ERROR: (
        "The diagnostic harness truncated its own probe (unexpected signal "
        "or exit). Re-run this diagnostic unchanged; never attribute any "
        "component from this run."
    ),
    CLASS_UNRESOLVED: (
        "Campaign allowance reached while the turn stayed live-but-uncertain. "
        "Recorded evidence (snapshots, audit, internal OpenCode status) plus "
        "one optional resumable experiment with the configurable "
        "--campaign-allowance-s raised. INCONCLUSIVE is not a product FAIL."
    ),
    CLASS_BOUNDARY_LOCALIZED: (
        "The turn did not deliver and no provider/server error was proven: "
        "BOUNDARY LOCALIZED / ROOT UNKNOWN. Use the last observed boundary "
        "as the starting point; do not assign an unproven root cause."
    ),
    CLASS_STAGE_UNOBSERVED: (
        "The turn produced NO observed stage marker. Use the captured "
        "thread stacks/task states and OpenCode request audit to identify "
        "the exact in-flight call; rerun with deeper stage hooks only if "
        "stacks are not enough. Do not assign an upstream cause."
    ),
    CLASS_DELIVERED: (
        "No isolated fix is indicated: the scenario delivered in the "
        "observable window with no per-answer cutoff; attach timing "
        "evidence to the broad repair analysis (kodmial/aa#417)."
    ),
}


def _run_url() -> str:
    """Actions run URL (no credentials in URL)."""
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY", "kodmial/aa")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    return f"{server}/{repository}/actions/runs/{run_id}".strip()


def _summary_banner(classification: str) -> str:
    """Prominent, non-deceptive one-line outcome for step summaries."""
    if classification == CLASS_DELIVERED:
        return "> **outcome: delivered — full scenario #2 answer delivered**"
    if classification in INCONCLUSIVE_CLASSIFICATIONS:
        return (
            f"> **outcome: INCONCLUSIVE ({classification}) — NOT a success; "
            "this run did not reach a truthful root-cause verdict.**"
        )
    return f"> **outcome: failure ({classification})**"


def build_final_report(meta: dict[str, Any], probe_a: ProbeEvidence, probe_b: ProbeEvidence) -> str:
    """Compact actionable truthful report (privacy-safe, one screen)."""
    classification = classify_cause(probe_a, probe_b)
    next_fix = _NEXT_FIX.get(classification, "")
    inconclusive = classification in INCONCLUSIVE_CLASSIFICATIONS
    lines = [
        _summary_banner(classification),
        "",
        "Single scenario #2 (`substantive-drinking`) diagnostic; NOT a Gate C "
        "PASS/FAIL, no full 18-case rerun and no product criterion changed.",
        "No per-answer wall-clock cutoff exists: a model call continuing "
        "past 45/75/130/300/900 s is observed, not killed; observation ends "
        "only on delivery, a proven error, sustained local-server "
        "non-responsiveness or the campaign physical allowance.",
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
    if inconclusive:
        lines.append("- verdict: INCONCLUSIVE — this is NOT a success")
    boundary_b = probe_b.last_boundary_stage
    boundary_a = probe_a.last_boundary_stage
    if classification == CLASS_STAGE_UNOBSERVED:
        lines.append("- stage=unobserved (explicit: no stage markers arrived)")
    else:
        if boundary_a:
            lines.append(f"- probe A last boundary: `{boundary_a}` (boundary only)")
        if boundary_b:
            lines.append(f"- probe B last boundary: `{boundary_b}` (boundary only)")
        hint = _STAGE_HINTS.get(boundary_b, "")
        if hint:
            lines.append(f"- boundary hint: {hint}")
    if probe_b.last_completed_stage:
        lines.append(f"- last completed stage: `{probe_b.last_completed_stage}`")
    if probe_b.last_real_event:
        lines.append(f"- last confirmed REAL activity: `{probe_b.last_real_event}`")
    if classification == CLASS_UNRESOLVED:
        lines.append(
            "- instrumentation gap: the run-recorded evidence cannot "
            "discriminate a live-but-slow inference from a hung request at "
            "the observed boundary, so no component is attributed"
        )
    if classification == CLASS_HARNESS_ERROR:
        lines.append(
            "- evidence integrity: a probe was truncated by the diagnostic "
            "harness itself; NO stage-level or component-level cause is "
            "proven by this run"
        )
    lines.append(f"- snapshots captured: A={probe_a.snapshots}, B={probe_b.snapshots}")
    lines.append(
        f"- child killed by supervisor: A={str(probe_a.killed).lower()}, "
        f"B={str(probe_b.killed).lower()}"
    )
    budget = meta.get("effective_campaign_allowance_s")
    if isinstance(budget, (int, float)) and budget > 0:
        lines.append(f"- campaign allowance applied: {round(float(budget), 1)}s")
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
        "exited_by_signal": evidence.exit_signal,
        "terminated_at_campaign_end": evidence.terminated_at_campaign_end,
        "forced_exit": evidence.forced_exit,
        "server_nonresponsive_sustained": evidence.server_nonresponsive_sustained,
        "killed": evidence.killed,
        "last_stage": evidence.last_stage,
        "active_stage": evidence.active_stage,
        "active_stage_ts": evidence.active_stage_ts,
        "last_completed_stage": evidence.last_completed_stage,
        "last_completed_stage_ms": evidence.last_completed_stage_elapsed_ms,
        "last_real_event": evidence.last_real_event,
        "delivered": evidence.delivered,
        "delivered_elapsed_ms": evidence.delivered_elapsed_ms,
        "fail_category": evidence.fail_category,
        "server_ready_ms": evidence.server_ready_ms,
        "model_request_total_ms": evidence.model_elapsed_ms,
        "snapshots": evidence.snapshots,
        "timeline": list(evidence.timeline)[:24],
    }


# ---------------------------------------------------------------------------
# kodmial/aa#553: parent-side bounded sequential A/B supervision
# ---------------------------------------------------------------------------


def _ab_arm_fallback(evidence: ProbeEvidence, arm: str) -> dict[str, Any]:
    """Explicit unknown arm evidence when the child emitted no A/B result.

    Nothing is inferred: absent telemetry stays missing and the arm can never
    contribute an A/B COMPLETE or a product PASS.
    """
    return {
        "arm": arm,
        "probe": evidence.probe,
        "mode": MODE_ONE_SHOT_AB,
        "status": str(evidence.status or "unknown")[:48],
        "delivered": bool(evidence.delivered),
        "timed_out": bool(evidence.terminated_at_campaign_end),
        "typed_failure": str(evidence.fail_category or "")[:64],
        "env_flag_raw": "",
        "expected_env_flag": expected_ab_env_flag(arm),
        "effective_flag": None,
        "scenario": ab_scenario_identity(),
        "budget": _ab_budget(),
        "preflight_elapsed_ms": None,
        "live_elapsed_ms": evidence.delivered_elapsed_ms,
        "delivery_ms": evidence.delivered_elapsed_ms,
        "retrieval": {},
        "attempts": {},
        "telemetry": {},
        "product": {},
        "provider_429": False,
        "active_backoff": False,
    }


def _ab_status_banner(status: str) -> str:
    """Non-deceptive one-line A/B outcome; never a qualification verdict."""
    if status == AB_STATUS_COMPLETE:
        return (
            "> **outcome: COMPLETE — bounded A/B evidence only; NOT a "
            "qualification verdict, NOT a product PASS and no speedup is "
            "inferred.**"
        )
    if status == AB_STATUS_DEDUPLICATED:
        return (
            "> **outcome: DEDUPLICATED — this exact SHA/config already has a "
            "COMPLETE A/B result; no provider run was repeated.**"
        )
    if status == AB_STATUS_BLOCKED:
        return (
            "> **outcome: BLOCKED — an arm failed or a provider 429/backoff/"
            "timeout stopped the A/B; NOT a success.**"
        )
    return (
        f"> **outcome: {status} — incomplete/uncertain evidence; NOT a "
        "success and no product PASS.**"
    )


def _ab_arm_report_lines(arm: dict[str, Any]) -> list[str]:
    """Privacy-safe observed evidence for one arm (no prompt/reply text)."""
    retrieval = arm.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, dict) else {}
    attempts = arm.get("attempts")
    attempts = attempts if isinstance(attempts, dict) else {}
    telemetry = arm.get("telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    product = arm.get("product")
    product = product if isinstance(product, dict) else {}
    by_agent = attempts.get("by_agent")
    by_agent = by_agent if isinstance(by_agent, dict) else {}
    agent_counts = ", ".join(
        f"{str(name)[:32]}={int((value or {}).get('attempts', 0))}"
        for name, value in sorted(by_agent.items())
        if isinstance(value, dict)
    )
    lines = [
        f"### Arm `{arm.get('arm', 'unknown')}`",
        f"- status: `{arm.get('status', 'unknown')}`",
        f"- env_flag_raw: `{arm.get('env_flag_raw', '')}`",
        f"- expected_env_flag: `{arm.get('expected_env_flag', '')}`",
        f"- effective_flag: `{arm.get('effective_flag')}`",
        f"- delivered: `{str(bool(arm.get('delivered'))).lower()}`",
        f"- timed_out: `{str(bool(arm.get('timed_out'))).lower()}`",
    ]
    for key in ("preflight_elapsed_ms", "live_elapsed_ms", "delivery_ms"):
        if isinstance(arm.get(key), (int, float)):
            lines.append(f"- {key}: {arm[key]}")
    if arm.get("typed_failure"):
        lines.append(f"- typed_failure: `{str(arm['typed_failure'])[:64]}`")
    lines.append(f"- retrieval_mode: `{retrieval.get('retrieval_mode', 'unobserved')}`")
    lines.append(f"- fastpath_attempted: `{retrieval.get('fastpath_attempted', 'unobserved')}`")
    lines.append(f"- fallback_reason: `{retrieval.get('fallback_reason', '')}`")
    lines.append(f"- coverage_status: `{retrieval.get('coverage_status', 'unobserved')}`")
    if "LLM_call_count" in retrieval:
        lines.append(f"- retrieval_llm_call_count: {retrieval['LLM_call_count']}")
    if "calls_observed" in retrieval:
        lines.append(f"- retrieval_calls_observed: {retrieval['calls_observed']}")
    lines.append(f"- model_http_attempts: {attempts.get('model_http_attempts', 'unobserved')}")
    lines.append(f"- attempts_by_agent: {agent_counts or 'unobserved'}")
    for key in (
        "planner_latency_ms",
        "retrieval_latency_ms",
        "answer_latency_ms",
        "verifier_latency_ms",
        "total_latency_ms",
    ):
        if isinstance(telemetry.get(key), (int, float)):
            lines.append(f"- {key}: {telemetry[key]}")
    for key in ("planner_outcome", "retrieval_outcome", "answer_outcome", "verifier_outcome"):
        if telemetry.get(key) not in (None, ""):
            lines.append(f"- {key}: `{str(telemetry[key])[:64]}`")
    for key in ("verified_book_units", "adequacy_verdict", "all_required_supported"):
        if key in telemetry:
            lines.append(f"- {key}: `{str(telemetry[key])[:64]}`")
    lines.append(f"- product_outcome: `{str(product.get('product_outcome', 'unknown'))[:64]}`")
    lines.append(f"- grounded_substantive: `{product.get('grounded_substantive')}`")
    if isinstance(product.get("reply_len"), int):
        lines.append(f"- reply_len: {product['reply_len']}")
    if attempts.get("failure_categories"):
        categories = ", ".join(str(item)[:48] for item in attempts["failure_categories"][:8])
        lines.append(f"- attempt_failure_categories: `{categories}`")
    missing = arm.get("missing_fields")
    if isinstance(missing, list) and missing:
        lines.append("- missing_fields: `" + ", ".join(str(item)[:48] for item in missing) + "`")
    else:
        lines.append("- missing_fields: none")
    return lines


def build_ab_report(
    meta: dict[str, Any],
    arms: list[dict[str, Any]],
    status: str,
    *,
    second_arm_skipped_reason: str = "",
) -> str:
    """Compact truthful A/B report; evidence only, never qualification."""
    scenario: dict[str, Any] = dict(meta.get("scenario") or {})
    config: dict[str, Any] = dict(meta.get("config") or {})
    budget: dict[str, Any] = dict(meta.get("budget") or {})
    lines = [
        _ab_status_banner(status),
        "",
        "Bounded real-book one-shot retrieval A/B (`kodmial/aa#553`). This is "
        "validation wiring only: it does not change production semantics, the "
        "runtime flag default, Gate C acceptance rules, qualification status "
        "or any repair/recovery cap.",
        f"Run: `{meta.get('run_url', '')}` (exact SHA `{meta.get('main_sha', '')}`)",
        f"- mode: `{meta.get('mode', MODE_ONE_SHOT_AB)}`",
        f"- scenario_family: `{scenario.get('family', 'unknown')}`",
        f"- scenario_source: `{scenario.get('source', 'unknown')}`",
        f"- prompt_sha256: `{scenario.get('prompt_sha256', '')}`",
        f"- config_digest: `{config.get('config_digest', '')}`",
        f"- primary_model: `{config.get('primary_model', '')}`",
        f"- fallback_model: `{config.get('fallback_model', '')}`",
        f"- live_budget_s: {budget.get('live_budget_s', '')}",
        f"- preflight_budget_s: {budget.get('preflight_budget_s', '')}",
        f"- cleanup_reserve_s: {budget.get('cleanup_reserve_s', '')}",
        f"- total_real_model_budget_cap_s: {budget.get('total_real_model_budget_cap_s', '')}",
        "",
    ]
    for arm in arms:
        lines.extend(_ab_arm_report_lines(arm))
        lines.append("")
    if second_arm_skipped_reason:
        lines.append(f"- second_arm_skipped_reason: `{second_arm_skipped_reason}`")
        lines.append("")
    lines.extend(
        [
            "### Decision boundary",
            "- A/B is evidence for a controller decision only; it is NOT an "
            "authoritative qualification and cannot admit #538/#514/#7.",
            "- No speedup, no correctness/grounding/safety PASS and no product "
            "verdict is inferred from missing or incomplete telemetry.",
            "- The production flag default remains OFF; only an explicit "
            "controller admission may enable it in the trusted qualification path.",
        ]
    )
    return "\n".join(lines)


def ab_already_complete(
    publisher: IssueProgressPublisher,
    main_sha: str,
    config_digest: str,
) -> bool:
    """Whether this exact SHA/config already has a COMPLETE A/B comment."""
    comments = publisher.list_comments()
    if not comments:
        return False
    marker = ab_complete_marker(main_sha, config_digest)
    return any(
        isinstance(comment.get("body"), str) and marker in str(comment["body"])
        for comment in comments
    )


def _ab_exit_code(status: str) -> int:
    """Truthful process exit code for the A/B status."""
    if status in (AB_STATUS_COMPLETE, AB_STATUS_DEDUPLICATED):
        return EXIT_OK
    if status == AB_STATUS_BLOCKED:
        return EXIT_BOOTSTRAP_FAILED
    return EXIT_INCONCLUSIVE


def _ab_outcome(status: str) -> str:
    return {
        AB_STATUS_COMPLETE: "complete",
        AB_STATUS_DEDUPLICATED: "deduplicated",
        AB_STATUS_INCOMPLETE: "incomplete",
        AB_STATUS_INCONCLUSIVE: "inconclusive",
        AB_STATUS_BLOCKED: "blocked",
    }.get(status, "unknown")


def _write_ab_evidence(
    out_dir: Path,
    meta: dict[str, Any],
    arms: list[dict[str, Any]],
    status: str,
    *,
    second_arm_skipped_reason: str = "",
) -> tuple[dict[str, Any], str]:
    """Persist result.json + final-report.md and return (result, report)."""
    exit_code = _ab_exit_code(status)
    missing_fields = {
        str(arm.get("arm", "unknown")): list(arm.get("missing_fields") or []) for arm in arms
    }
    report = build_ab_report(
        meta,
        arms,
        status,
        second_arm_skipped_reason=second_arm_skipped_reason,
    )
    (out_dir / "final-report.md").write_text(report + "\n", encoding="utf-8")
    result = {
        "schema": AB_SCHEMA_VERSION,
        "mode": MODE_ONE_SHOT_AB,
        "status": status,
        "outcome": _ab_outcome(status),
        "exit_code": exit_code,
        "main_sha": meta.get("main_sha", ""),
        "run_id": meta.get("run_id", ""),
        "run_url": meta.get("run_url", ""),
        "target_issue": AB_ISSUE,
        "scenario": meta.get("scenario", {}),
        "config": meta.get("config", {}),
        "budget": meta.get("budget", {}),
        "campaign_allowance_s": meta.get("effective_campaign_allowance_s"),
        "per_answer_wall_clock_cutoff_s": None,
        "second_arm_skipped_reason": second_arm_skipped_reason,
        "arms": arms,
        "missing_fields": missing_fields,
        "diagnostic_note": (
            "Bounded validation-only real-book A/B; not a qualification, not "
            "a Gate C PASS/FAIL, not a product PASS and never a speedup claim."
        ),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result, report


def _parent_ab_main(
    args: argparse.Namespace,
    out_dir: Path,
    allowance: float,
    token: str,
) -> int:
    """Run OFF then ON sequentially under the existing bounded canary budget."""
    budget = _ab_budget(
        float(getattr(args, "ab_live_budget_s", DEFAULT_AB_LIVE_BUDGET_S)),
        float(getattr(args, "ab_preflight_budget_s", DEFAULT_AB_PREFLIGHT_BUDGET_S)),
        float(getattr(args, "ab_cleanup_reserve_s", DEFAULT_AB_CLEANUP_RESERVE_S)),
    )
    config = ab_config_identity(
        os.environ.get("OPENCODE_MODEL", ""),
        os.environ.get("OPENCODE_FALLBACK_MODEL", ""),
        live_budget_s=budget["live_budget_s"],
        preflight_budget_s=budget["preflight_budget_s"],
        cleanup_reserve_s=budget["cleanup_reserve_s"],
    )
    scenario = ab_scenario_identity()
    meta: dict[str, Any] = {
        "schema": AB_SCHEMA_VERSION,
        "mode": MODE_ONE_SHOT_AB,
        "main_sha": args.main_sha or git_head_sha(),
        "run_id": str(args.run_id),
        "run_url": _run_url(),
        "scenario": scenario,
        "config": config,
        "budget": budget,
        "campaign_allowance_s": float(args.campaign_allowance_s),
        "effective_campaign_allowance_s": allowance,
        "finalisation_reserve_s": float(args.finalisation_reserve_s),
        "job_timeout_minutes": float(args.job_timeout_minutes),
        "per_answer_wall_clock_cutoff_s": None,
    }
    (out_dir / "run-meta.json").write_text(
        json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    sink = EventSink(out_dir / "supervisor-events.jsonl", echo=False)
    sink.emit(
        "supervisor-start",
        mode=MODE_ONE_SHOT_AB,
        run_id=meta["run_id"],
        sha=meta["main_sha"],
        config_digest=config["config_digest"],
        effective_campaign_allowance_s=round(allowance, 1),
    )
    publisher = IssueProgressPublisher(
        args.owner,
        args.repo,
        int(getattr(args, "ab_issue", AB_ISSUE)),
        token,
        run_id=str(args.run_id),
        marker=ab_run_marker(str(args.run_id)),
    )
    arms: list[dict[str, Any]] = []
    skipped_reason = ""
    if ab_already_complete(publisher, str(meta["main_sha"]), config["config_digest"]):
        status = AB_STATUS_DEDUPLICATED
        sink.emit(
            "supervisor-ab-deduplicated",
            sha=meta["main_sha"],
            config_digest=config["config_digest"],
        )
    else:
        if token:
            publisher.publish(
                "One-shot retrieval A/B started",
                f"- run: `{meta['run_url']}`\n- sha: `{meta['main_sha']}`\n"
                f"- mode: `{MODE_ONE_SHOT_AB}`\n"
                f"- config_digest: `{config['config_digest']}`\n"
                f"- arms: `off` then `on` (sequential; bounded canary budget)",
            )
        campaign_deadline = time.monotonic() + allowance
        for arm in (AB_ARM_OFF, AB_ARM_ON):
            if arm == AB_ARM_ON:
                remaining = campaign_deadline - time.monotonic()
                required = (
                    MIN_AB_PREFLIGHT_S
                    + float(budget["live_budget_s"])
                    + float(budget["cleanup_reserve_s"])
                )
                sink.emit(
                    "supervisor-ab-budget-check",
                    next_arm=arm,
                    remaining_s=round(remaining, 1),
                    required_s=round(required, 1),
                )
                if remaining < required:
                    skipped_reason = "insufficient-job-budget"
                    break
            probe_label = PROBE_SCENARIO2_OFF if arm == AB_ARM_OFF else PROBE_SCENARIO2_ON
            sink.emit(
                "supervisor-ab-arm-start",
                arm=arm,
                probe=probe_label,
                env_flag=expected_ab_env_flag(arm),
            )
            evidence = _run_one_probe(
                ProbePlan(
                    probe=probe_label,
                    snapshot_cadence_s=float(args.snapshot_cadence_s),
                    max_snapshots=int(args.max_snapshots),
                    connect_timeout_s=float(args.probe_a_connect_timeout_s),
                    arm_env=ab_arm_environment(arm),
                    ab_mode=True,
                    ab_arm=arm,
                    ab_live_budget_s=float(budget["live_budget_s"]),
                    ab_preflight_budget_s=float(budget["preflight_budget_s"]),
                    ab_cleanup_reserve_s=float(budget["cleanup_reserve_s"]),
                ),
                out_dir=out_dir,
                sink=sink,
                publisher=publisher,
                campaign_deadline_s=campaign_deadline,
            )
            arm_result = (
                dict(evidence.ab_result)
                if isinstance(evidence.ab_result, dict) and evidence.ab_result
                else _ab_arm_fallback(evidence, arm)
            )
            arm_result["arm"] = arm
            arm_result["probe"] = probe_label
            arm_result["mode"] = MODE_ONE_SHOT_AB
            arm_result["expected_env_flag"] = expected_ab_env_flag(arm)
            arm_result["exit_code"] = evidence.exit_code
            arm_result["evidence_status"] = evidence.status
            arm_result["snapshots"] = evidence.snapshots
            arm_result["killed"] = evidence.killed
            arm_result["terminated_at_campaign_end"] = evidence.terminated_at_campaign_end
            arm_result["missing_fields"] = _ab_missing_fields(arm_result)
            arms.append(arm_result)
            stop, reason = _ab_should_stop_after_arm(arm_result)
            if stop:
                skipped_reason = reason
                break
        status = classify_ab_outcome(
            arms,
            second_arm_skipped_reason=skipped_reason,
        )
    result, report = _write_ab_evidence(
        out_dir,
        meta,
        arms,
        status,
        second_arm_skipped_reason=skipped_reason,
    )
    if token:
        body = report
        if status == AB_STATUS_COMPLETE:
            body = (
                ab_complete_marker(str(meta["main_sha"]), str(config["config_digest"]))
                + "\n"
                + report
            )
        publisher.publish(f"One-shot retrieval A/B {status}", body)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if step_summary:
        try:
            with Path(step_summary).open("a", encoding="utf-8") as stream:
                stream.write("## AA one-shot retrieval A/B (validation only)\n\n")
                stream.write(_ab_status_banner(status) + "\n\n```markdown\n")
                stream.write(report)
                stream.write("\n```\n")
        except OSError:
            pass
    sink.emit(
        "supervisor-report-published",
        mode=MODE_ONE_SHOT_AB,
        status=status,
        target_issue=AB_ISSUE,
    )
    print(
        json.dumps(
            {
                "mode": MODE_ONE_SHOT_AB,
                "status": status,
                "outcome": result["outcome"],
                "exit_code": result["exit_code"],
                "config_digest": config["config_digest"],
                "final_report": str(out_dir / "final-report.md"),
                "artifact": (f"aa-gate-c-scenario2-diagnostic-{meta['run_id']}-one-shot-ab"),
            }
        ),
        flush=True,
    )
    return int(result["exit_code"])


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
    allowance = effective_campaign_allowance_s(
        campaign_allowance_s=float(args.campaign_allowance_s),
        finalisation_reserve_s=float(args.finalisation_reserve_s),
        job_timeout_minutes=float(args.job_timeout_minutes),
        job_started_epoch=float(args.job_started_epoch),
    )
    if str(getattr(args, "mode", MODE_SCENARIO2)) == MODE_ONE_SHOT_AB:
        return _parent_ab_main(
            args,
            out_dir,
            allowance,
            (os.environ.get("GITHUB_TOKEN", "") or "").strip(),
        )
    meta = {
        "schema": SCHEMA_VERSION,
        "main_sha": args.main_sha or git_head_sha(),
        "run_id": str(args.run_id),
        "target_issue": int(args.issue),
        "target_owner": args.owner,
        "target_repo": args.repo,
        "run_url": _run_url(),
        # kodmial/aa#424: no per-answer cutoff exists in this diagnostic.
        "per_answer_wall_clock_cutoff_s": None,
        "campaign_allowance_s": float(args.campaign_allowance_s),
        "effective_campaign_allowance_s": allowance,
        "finalisation_reserve_s": float(args.finalisation_reserve_s),
        "job_timeout_minutes": float(args.job_timeout_minutes),
        "snapshot_cadence_s": float(args.snapshot_cadence_s),
        "max_snapshots": int(args.max_snapshots),
    }
    (out_dir / "run-meta.json").write_text(
        json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    sink = EventSink(out_dir / "supervisor-events.jsonl", echo=False)
    sink.emit(
        "supervisor-start",
        run_id=meta["run_id"],
        sha=meta["main_sha"],
        policy="no-per-answer-cutoff",
        effective_campaign_allowance_s=round(allowance, 1),
    )

    token = (os.environ.get("GITHUB_TOKEN", "") or "").strip()
    publisher = IssueProgressPublisher(
        args.owner, args.repo, args.issue, token, run_id=str(args.run_id)
    )
    if token:
        publisher.publish(
            "Scenario #2 diagnostic started",
            f"- run: `{meta['run_url']}`\n- sha: `{meta['main_sha']}`\n"
            f"- campaign allowance: {round(allowance)}s (no per-answer cutoff)",
        )

    campaign_deadline = time.monotonic() + allowance
    probe_a = ProbeEvidence(probe=PROBE_DIRECT_MODEL, status="not-run")
    probe_b = ProbeEvidence(probe=PROBE_SCENARIO2, status="not-run")
    os.environ["SCENARIO2_DIAG_PROBE_A_PORT"] = str(args.probe_a_port)
    os.environ["SCENARIO2_DIAG_PROBE_A_BOOTSTRAP_TIMEOUT_S"] = str(args.probe_a_bootstrap_timeout_s)
    os.environ["SCENARIO2_DIAG_PROBE_B_BOOTSTRAP_TIMEOUT_S"] = str(args.probe_b_bootstrap_timeout_s)
    try:
        if args.probe in ("both", PROBE_DIRECT_MODEL):
            probe_a = _run_one_probe(
                ProbePlan(
                    probe=PROBE_DIRECT_MODEL,
                    snapshot_cadence_s=float(args.snapshot_cadence_s),
                    max_snapshots=int(args.max_snapshots),
                    connect_timeout_s=float(args.probe_a_connect_timeout_s),
                ),
                out_dir=out_dir,
                sink=sink,
                publisher=publisher,
                campaign_deadline_s=campaign_deadline,
            )
        if args.probe in ("both", PROBE_SCENARIO2):
            sink.emit(
                "supervisor-campaign-budget-check",
                remaining_s=round(campaign_deadline - time.monotonic(), 1),
            )
            probe_b = _run_one_probe(
                ProbePlan(
                    probe=PROBE_SCENARIO2,
                    snapshot_cadence_s=float(args.snapshot_cadence_s),
                    max_snapshots=int(args.max_snapshots),
                    connect_timeout_s=float(args.probe_a_connect_timeout_s),
                ),
                out_dir=out_dir,
                sink=sink,
                publisher=publisher,
                campaign_deadline_s=campaign_deadline,
            )
    finally:
        sink.emit("supervisor-stray-check", opencode_processes_leftover=count_opencode_processes())

    classification = classify_cause(probe_a, probe_b)
    exit_code = _EXIT_BY_CLASSIFICATION.get(classification, EXIT_INCONCLUSIVE)
    outcome = {
        CLASS_DELIVERED: "delivered",
        CLASS_PROVIDER_ERROR_PROVEN: "failure",
        CLASS_SERVER_ERROR_PROVEN: "failure",
        CLASS_HARNESS_ERROR: "failure",
    }.get(classification, "inconclusive")
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
        "classification": classification,
        "next_fix": _NEXT_FIX.get(classification, ""),
        "exit_code": exit_code,
        "outcome": outcome,
        "campaign_allowance_s": meta["effective_campaign_allowance_s"],
        "per_answer_wall_clock_cutoff_s": None,
        "diagnostic_note": (
            "Diagnostic observation only; not a product qualification "
            "timeout and not a Gate C PASS/FAIL. INCONCLUSIVE is never a "
            "product FAIL and is never reported as a success."
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
                stream.write("## AA Gate C scenario2 diagnostic\n\n")
                stream.write(_summary_banner(classification) + "\n\n```markdown\n")
                stream.write(report)
                stream.write("\n```\n")
        except OSError:
            pass
    sink.emit("supervisor-report-published", target_issue=meta["target_issue"])
    print(
        json.dumps(
            {
                "outcome": outcome,
                "exit_code": exit_code,
                "classification": classification,
                "final_report": str(out_dir / "final-report.md"),
                "artifact": f"aa-gate-c-scenario2-diagnostic-{meta['run_id']}-scenario2",
            }
        ),
        flush=True,
    )
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """CLI surface for the parent mode (child params travel via env)."""
    parser = argparse.ArgumentParser(
        description=(
            "Isolated Gate C scenario #2 diagnostic; progress/liveness-driven "
            "observation with one finite campaign allowance (no per-answer "
            "cutoff, kodmial/aa#424)."
        )
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=(MODE_SCENARIO2, MODE_ONE_SHOT_AB),
        default=MODE_SCENARIO2,
        help=(
            "scenario2 (existing diagnostic, default) or the #553 bounded "
            "validation-only real-book one-shot retrieval A/B."
        ),
    )
    parser.add_argument(
        "--ab-issue",
        type=int,
        default=AB_ISSUE,
        help="Issue that receives the bounded A/B evidence comment.",
    )
    parser.add_argument(
        "--ab-live-budget-s",
        type=float,
        default=DEFAULT_AB_LIVE_BUDGET_S,
        help="Per-arm live canary budget (default 300s).",
    )
    parser.add_argument(
        "--ab-preflight-budget-s",
        type=float,
        default=DEFAULT_AB_PREFLIGHT_BUDGET_S,
        help="Per-arm preflight canary budget (default 600s).",
    )
    parser.add_argument(
        "--ab-cleanup-reserve-s",
        type=float,
        default=DEFAULT_AB_CLEANUP_RESERVE_S,
        help="Per-arm cleanup reserve kept inside the live budget.",
    )
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
    # kodmial/aa#424: the ONE finite campaign ceiling + finalisation reserve.
    parser.add_argument(
        "--campaign-allowance-s",
        type=float,
        default=DEFAULT_CAMPAIGN_ALLOWANCE_S,
        help=(
            "Job-level physical ceiling shared by both probes (default "
            "1800s, mirroring the AA qualification's real-progress "
            "scenario allowance); never a per-answer timeout."
        ),
    )
    parser.add_argument(
        "--finalisation-reserve-s",
        type=float,
        default=DEFAULT_FINALISATION_RESERVE_S,
        help="Time kept for artifacts/upload inside the job budget.",
    )
    parser.add_argument(
        "--job-timeout-minutes",
        type=float,
        default=0.0,
        help="Actions job timeout, used ONLY with --job-started-epoch.",
    )
    parser.add_argument(
        "--job-started-epoch",
        type=float,
        default=0.0,
        help="Epoch seconds when the Actions job started (job reconciliation).",
    )
    parser.add_argument(
        "--snapshot-cadence-s",
        type=float,
        default=DEFAULT_SNAPSHOT_CADENCE_S,
        help="Periodic incremental snapshot cadence (bounded log volume).",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=DEFAULT_MAX_SNAPSHOTS,
        help="Bounded snapshot count per probe (log-volume cap).",
    )
    parser.add_argument("--probe-a-port", default=DEFAULT_PROBE_A_PORT)
    parser.add_argument(
        "--probe-a-connect-timeout-s",
        type=float,
        default=DEFAULT_PROBE_A_CONNECT_TIMEOUT_S,
        help=(
            "Explicitly-typed loopback TCP-connect bound only; NEVER a response/answer deadline."
        ),
    )
    parser.add_argument(
        "--probe-a-bootstrap-timeout-s",
        type=float,
        default=DEFAULT_PROBE_A_BOOTSTRAP_TIMEOUT_S,
    )
    parser.add_argument(
        "--probe-b-bootstrap-timeout-s",
        type=float,
        default=DEFAULT_PROBE_B_BOOTSTRAP_TIMEOUT_S,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: dispatch to the child or the parent mode."""
    args = build_parser().parse_args(argv)
    if args.child:
        child = child_argv_from_env()
        if child["probe"] not in (PROBE_DIRECT_MODEL, *SCENARIO2_PROBES):
            return EXIT_REPORT_FAILED
        return _child_main(child)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
