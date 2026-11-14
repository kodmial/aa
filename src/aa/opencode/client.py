"""Local OpenCode invocation boundary.

The production runtime pins OpenCode 2.2.6 and speaks its v2 HTTP API over
loopback. The v1 surface is retained only for historical fixtures and
qualification doubles:

- ``GET /api/info`` (v2) or ``GET /global/health`` (v1) for readiness;
- ``POST /api/session`` / ``GET /api/session/{id}`` /
  ``DELETE /api/session/{id}`` for session lifecycle;
- ``POST /api/session/{id}/agent`` and ``/model`` to pin the wire
  selectors for subsequent provider turns;
- ``POST /api/session/{id}/prompt`` to *enqueue* one user input. The v2
  response is durable-admission evidence, never proof that an assistant
  reply exists;
- ``POST /api/experimental/session/{id}/wait`` to await the actual agent
  loop becoming idle, then paged ``GET /api/session/{id}/message`` to pull
  the completed assistant message;
- ``POST /api/session/{id}/interrupt`` for cancelling in-flight work.

Only the standard library is used (``urllib`` executed in a worker thread),
so the worker gains no second LLM client and no HTTP vendor dependency.
Model/provider (Zen) configuration stays on the OpenCode side: the worker
either omits the model (server default applies) or forwards the opaque
``OPENCODE_MODEL`` ``provider/model`` pointer. Basic auth uses the
``opencode`` username and the managed ``OPENCODE_PASSWORD``; credentials are
never logged and prompt text is never logged.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from aa.conversation.turn_causality import clear_turn_causality as _causality_clear
from aa.conversation.turn_causality import stamp as _causality_stamp
from aa.opencode.attempt_audit import AttemptLedger
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeError,
    OpenCodeSessionNotFoundError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
    classify_http_status,
    classify_provider_error,
    classify_structured_error,
)

logger = logging.getLogger("aa.opencode.client")

# A v2 prompt is admitted asynchronously. The completion loop re-checks the
# session after ``/wait`` returns, because the wait route only observes the
# agent loop: a prompt admitted a moment earlier may not have marked the loop
# busy yet. The loop therefore waits for the turn's *own* terminal idle
# message instead of treating a successful enqueue or immediate wait return as
# a reply. A caller-supplied timeout bounds the whole wait; the fallback keeps
# the loop finite when no timeout was configured. Polling happens inside the
# blocking worker thread and is therefore cancellable via the attempt
# supervisor (cancellation also issues the native interrupt).
_V2_COMPLETION_POLL_SECONDS = 0.05
_V2_COMPLETION_FALLBACK_TIMEOUT_SECONDS = 600.0


def _v2_budget_deadline(timeout: float | None) -> float:
    """Resolve one whole-exchange deadline (caller timeout or finite fallback)."""
    budget = _V2_COMPLETION_FALLBACK_TIMEOUT_SECONDS if timeout is None else timeout
    return time.monotonic() + max(0.001, budget)


def _v2_remaining(deadline: float) -> float:
    """Return the remaining whole-exchange budget or raise a timeout."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OpenCodeTimeoutError("opencode turn completion was not observed before the deadline")
    return remaining


@dataclass(frozen=True)
class HealthInfo:
    """Result of the readiness probe (``GET /api/info`` on v2)."""

    healthy: bool
    version: str = ""


@dataclass(frozen=True)
class SessionInfo:
    """An OpenCode session identity (opaque to the worker)."""

    id: str
    title: str = ""


@dataclass(frozen=True)
class ChatMessage:
    """One session message with extracted plain text."""

    role: str
    text: str = ""


def model_payload(hint: str) -> dict[str, str] | None:
    """Translate an ``OPENCODE_MODEL`` hint into an OpenCode v1 model payload.

    The hint is an opaque ``provider/model`` pointer owned by the OpenCode
    side. An empty hint returns ``None`` (the server default, i.e. the
    Zen-configured model, applies). A malformed non-empty hint raises a
    deterministic error so misconfiguration surfaces immediately.
    """
    parsed = _split_model_hint(hint)
    if parsed is None:
        return None
    provider, model = parsed
    return {"providerID": provider, "modelID": model}


def model_ref_payload(hint: str) -> dict[str, str] | None:
    """Translate an ``OPENCODE_MODEL`` hint into an OpenCode v2 ``Model.Ref``."""
    parsed = _split_model_hint(hint)
    if parsed is None:
        return None
    provider, model = parsed
    return {"providerID": provider, "id": model}


def _split_model_hint(hint: str) -> tuple[str, str] | None:
    if not hint:
        return None
    if "/" in hint:
        provider, _, model = hint.partition("/")
        if provider and model:
            return provider, model
    raise OpenCodeDeterministicError("malformed model hint; expected 'provider/model'")


def _extract_text(parts: object) -> str:
    """Extract concatenated plain text from OpenCode message parts."""
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)
            continue
        data = part.get("data")
        if isinstance(data, dict):
            nested = data.get("text")
            if isinstance(nested, str) and nested:
                chunks.append(nested)
    return "\n".join(chunks)


def _extract_v2_content(entry: object) -> str:
    """Extract visible text from one v2 assistant message ``content`` list."""
    if not isinstance(entry, dict):
        return ""
    content = entry.get("content")
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)
    return "\n".join(chunks)


def _v2_system_preamble(system: str, text: str) -> str:
    """Combine native v1 system text with the v2 user prompt.

    The v2 prompt request has no ``system`` field. The system instructions
    therefore travel first in the same user input, separated from the user
    text by a blank line so the model can still distinguish policy from
    conversation content.
    """
    instructions = system.strip()
    if not instructions:
        return text
    return f"{instructions}\n\n{text}"


def _v2_structured_prompt(text: str, *, system: str, schema: dict[str, object]) -> str:
    """Build the strict JSON-only prompt used in place of v1 native ``format``.

    OpenCode 2.2.6 accepts no ``format`` field on the v2 prompt route (extra
    keys are silently ignored). The schema is therefore supplied as an
    explicit instruction and the returned assistant text is parsed strictly;
    callers still Pydantic-validate the object.
    """
    schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    instruction = (
        "Return only one JSON object that conforms to the JSON Schema below. "
        "Do not wrap it in Markdown fences and do not add commentary or any "
        "text outside the JSON object.\n"
        "JSON Schema:\n"
        f"{schema_text}"
    )
    base = _v2_system_preamble(system, text)
    return f"{base}\n\n{instruction}"


def _parse_v2_json_object(text: str) -> dict[str, object] | None:
    """Strictly parse one v2 structured reply object.

    Accepts either a bare JSON object or one fenced JSON block. Free-form
    prose around the object is rejected so a caller never treats an
    accidentally embedded fragment as a complete structured response.
    """
    candidate = (text or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 2 and lines[0].strip().lower() in {"```", "```json"}:
            candidate = "\n".join(lines[1:])
            if candidate.rstrip().endswith("```"):
                candidate = candidate.rstrip()[:-3]
            candidate = candidate.strip()
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


def _extract_structured(payload: object) -> dict[str, object] | None:
    """Extract the native structured-output object from a prompt response."""
    if not isinstance(payload, dict):
        return None
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    for key in ("structured_output", "structured", "structuredOutput"):
        value = info.get(key)
        if isinstance(value, dict):
            return dict(value)
    return None


class OpenCodeClient(ABC):
    """Interface for the local OpenCode HTTP boundary (fakeable for tests)."""

    @property
    def request_latency_audit(self) -> tuple[dict[str, object], ...]:
        """Return privacy-safe local OpenCode HTTP timing evidence.

        The base client has no request audit; only the real HTTP boundary
        records bounded-censored local timing (kodmial/aa#417).
        """
        return ()

    @abstractmethod
    async def health(self) -> HealthInfo:
        """Probe runtime readiness."""
        raise NotImplementedError

    @abstractmethod
    async def create_session(self, title: str = "") -> SessionInfo:
        """Create a new OpenCode session and return its identity."""
        raise NotImplementedError

    @abstractmethod
    async def get_session(self, session_id: str) -> SessionInfo:
        """Fetch a session or raise :class:`OpenCodeSessionNotFoundError`."""
        raise NotImplementedError

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session; missing sessions raise not-found."""
        raise NotImplementedError

    @abstractmethod
    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
        system: str = "",
        format: dict[str, object] | None = None,
        audit_agent: str = "",
    ) -> str:
        """Send one user message and return the assistant text reply.

        ``system`` is delivered through OpenCode's native system layer,
        never flattened into ``parts`` text. ``format`` carries an
        optional native output-format object (for example a
        ``json_schema`` request). ``agent`` is the transport agent
        selector sent on the wire; ``audit_agent`` optionally records a
        distinct logical audit identity (for example the decoupled
        ``aa-verifier-v2`` verifier) while the wire omits the custom
        selector. When empty, the audit identity equals ``agent``.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
        system: str = "",
        schema: dict[str, object],
        retry_count: int = 2,
        audit_agent: str = "",
    ) -> dict[str, object]:
        """Send one message with native ``json_schema`` output and return it.

        The returned value is the structured-output object produced by
        OpenCode (``info.structured_output``/``info.structured``); callers
        Pydantic-validate it. No JSON text parsing happens here.
        ``audit_agent`` records a distinct logical audit identity while
        ``agent`` stays the wire transport selector (see
        :meth:`send_message`).
        """
        raise NotImplementedError

    @abstractmethod
    async def list_messages(self, session_id: str, *, limit: int = 50) -> list[ChatMessage]:
        """Return recent session messages (oldest first)."""
        raise NotImplementedError

    @abstractmethod
    async def abort(self, session_id: str) -> bool:
        """Abort in-flight work for a session."""
        raise NotImplementedError


class HttpOpenCodeClient(OpenCodeClient):
    """Real loopback HTTP client for ``opencode serve`` (stdlib only)."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float | None = None,
        protocol: str = "v1",
        password: str = "",
    ) -> None:
        if protocol not in ("v1", "v2"):
            raise ValueError("unknown OpenCode HTTP protocol")
        self._protocol = protocol
        self._password = password
        self._base_url = base_url.rstrip("/")
        if request_timeout is not None and request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        # ``request_timeout`` is an inactivity/connect bound for one blocking
        # socket operation, never an end-to-end response deadline. The
        # independently supervised qualification campaign owns the only
        # justified overall wall clock.
        self._request_timeout = request_timeout
        self._served_model_audit: list[dict[str, str]] = []
        self._token_usage_audit: list[dict[str, object]] = []
        # Session model pins persist on the server. Track the pins this client
        # applied so a later turn that omits the model cannot silently inherit
        # a stale pointer instead of the documented server default.
        self._v2_pinned_models: dict[str, str] = {}
        # Privacy-safe local transport timings: operation category, duration,
        # and success only. Never records prompts, responses, session ids, or
        # provider error text. Gate E uses this to distinguish model latency
        # from local session lifecycle overhead.
        self._request_latency_audit: list[dict[str, object]] = []
        # In-flight attempt ledger (kodmial/aa#431): attempt-start is written
        # BEFORE any HTTP worker thread is spawned so a hung call is
        # observable while it is still running, and terminal states are
        # published honestly (including late outcomes of cancelled calls).
        self._attempts = AttemptLedger()

    @property
    def base_url(self) -> str:
        """The configured loopback base URL."""
        return self._base_url

    def set_auth_password(self, password: str) -> None:
        """Bind managed v2 server authentication without emitting credentials."""
        self._password = password

    def _sync_request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
        timeout: float | None,
    ) -> object:
        """Perform one blocking HTTP request and return the decoded JSON.

        ``timeout`` is handed to the socket layer as an inactivity/connect
        bound. A server that keeps a request alive is never rejected by an
        arbitrary total response timer.
        """
        if self._protocol == "v2":
            if path == "/global/health":
                path = "/api/info"
            elif path.startswith("/api/"):
                pass
            elif path.startswith("/session"):
                path = "/api" + path
            else:
                raise OpenCodeDeterministicError("unsupported OpenCode v2 HTTP route")
        url = f"{self._base_url}{path}"
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        if self._protocol == "v2" and self._password:
            token = base64.b64encode(f"opencode:{self._password}".encode())
            headers["Authorization"] = "Basic " + token.decode("ascii")
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        # The inactivity/connect bound is enforced directly by the socket
        # layer on this blocking call; no asyncio timer wraps the response.
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            # The body may contain provider details; classify by status only.
            raise classify_http_status(exc.code) from exc
        if not raw:
            return None
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # The real v2 server answers known routes as JSON. An HTML SPA
            # fallback (the v1 routes do this) or any other non-JSON body is
            # a contract mismatch, never a usable protocol response.
            raise OpenCodeDeterministicError("opencode returned a non-JSON response") from exc
        if self._protocol == "v2" and isinstance(result, dict) and set(result) == {"data"}:
            # Unwrap only the single-field success envelopes. Paged
            # responses carry ``data`` plus ``cursor`` and must stay intact
            # so the caller can follow ``cursor.next``.
            return result["data"]
        return result

    def _worker_body(
        self,
        attempt_id: str,
        work: Callable[[], object],
        started_perf: float,
    ) -> object:
        """Blocking worker body with THREAD-SIDE honest fate recording.

        kodmial/aa#431: the worker thread records its own fate directly in
        the attempt ledger (a plain data structure), independent of the
        event loop. A coroutine abandoned by cancellation therefore still
        receives the worker's real terminal state as ``late_outcome``
        evidence, and the quiescence gate never confuses the asyncio-shim
        cancellation with the thread's actual lifecycle.
        """
        if not self._attempts.claim_worker_start(attempt_id):
            return None
        try:
            result = work()
        except Exception as exc:
            elapsed_ms = round(max(0.0, (time.perf_counter() - started_perf) * 1000.0), 1)
            failure = type(exc).__name__[:64]
            self._attempts.mark_worker_done(
                attempt_id, outcome="failed", latency_ms=elapsed_ms, failure_category=failure
            )
            self._attempts.record_late_outcome_if_cancelled(
                attempt_id, "failed", elapsed_ms, failure
            )
            raise
        elapse_net_ms = round(max(0.0, (time.perf_counter() - started_perf) * 1000.0), 1)
        self._attempts.mark_worker_done(attempt_id, outcome="completed", latency_ms=elapse_net_ms)
        self._attempts.record_late_outcome_if_cancelled(attempt_id, "completed", elapse_net_ms, "")
        return result

    @staticmethod
    def _request_operation(method: str, path: str, body: dict[str, object] | None) -> str:
        if method == "GET" and path in ("/global/health", "/api/info"):
            return "health"
        if method == "POST" and path in ("/session", "/api/session"):
            return "session-create"
        if path.startswith("/session/") and path.endswith("/message") and method == "POST":
            fmt = body.get("format") if isinstance(body, dict) else None
            return "message-structured" if isinstance(fmt, dict) else "message-text"
        if path.startswith("/session/") and path.endswith("/abort") and method == "POST":
            return "session-abort"
        if path.startswith("/api/session/") and path.endswith("/prompt") and method == "POST":
            return "message-text"
        if path.startswith("/api/session/") and path.endswith("/interrupt") and method == "POST":
            return "session-abort"
        if path.startswith("/api/session/") and path.endswith("/agent") and method == "POST":
            return "session-agent"
        if path.startswith("/api/session/") and path.endswith("/model") and method == "POST":
            return "session-model"
        if path.startswith("/api/experimental/session/") and path.endswith("/wait"):
            return "session-wait"
        if path.startswith("/api/session/") and "/message" in path and method == "GET":
            return "session-history"
        if path.startswith(("/session/", "/api/session/")) and method == "GET":
            return "session-get"
        if path.startswith(("/session/", "/api/session/")) and method == "DELETE":
            return "session-delete"
        return "other"

    def _open_request_row(
        self,
        *,
        operation: str,
        agent: str,
        audit_agent: str,
        model: str,
        fallback_model: str,
    ) -> tuple[str, dict[str, object]]:
        """Append the ``attempt-start`` audit row BEFORE the call runs.

        The row starts ``in-flight`` and is finalized in place by the same
        loop-side task; it never waits for the end of the blocking call, so
        a hung request always leaves observable evidence. Only operation
        category, bounded agent/model pointers and the causality stamp
        travel here — never prompt text, replies, session ids, credentials,
        or provider error bodies.
        """
        row: dict[str, object] = {
            "operation": operation,
            "state": "in-flight",
            "success": None,
            "latency_ms": 0.0,
            "agent": str(agent or "")[:128],
            "audit_agent": str(audit_agent or "")[:128],
            "model": str(model or "")[:128],
            "fallback_model": str(fallback_model or "")[:128],
        }
        row = _causality_stamp(dict(row))
        attempt_id = self._attempts.register_start(
            operation=operation,
            causality_id=str(row.get("causality_id", "")),
            stage=str(row.get("stage", "")),
            agent=str(row.get("agent", "")),
            audit_agent=str(row.get("audit_agent", "")),
            model=str(row.get("model", "")),
            fallback_model=str(row.get("fallback_model", "")),
        )
        row["attempt_id"] = attempt_id
        self._request_latency_audit.append(row)
        if len(self._request_latency_audit) > 2048:
            del self._request_latency_audit[: len(self._request_latency_audit) - 2048]
        return attempt_id, row

    def _finalize_request_row(
        self,
        attempt_id: str,
        row: dict[str, object],
        started_perf: float,
        *,
        success: bool | None,
        state: str,
        failure_category: str = "",
    ) -> None:
        """Transition the in-place audit row and ledger attempt to terminal."""
        elapsed_ms = round(max(0.0, (time.perf_counter() - started_perf) * 1000.0), 1)
        row["state"] = state
        row["latency_ms"] = elapsed_ms
        row["attempt_ended_monotonic"] = time.monotonic()
        if success is not None:
            row["success"] = success
        if failure_category:
            row["failure_category"] = str(failure_category)[:64]
        self._attempts.finish(
            attempt_id, state=state, latency_ms=elapsed_ms, failure_category=failure_category
        )

    @property
    def request_latency_audit(self) -> tuple[dict[str, object], ...]:
        """Return privacy-safe local OpenCode HTTP timing evidence."""
        return tuple(dict(item) for item in self._request_latency_audit)

    @property
    def attempt_audit(self) -> tuple[dict[str, object], ...]:
        """Return the privacy-safe in-flight attempt registry (kodmial/aa#431)."""
        return self._attempts.entries()

    def quiescence_proven(self, causality_id: str = "") -> bool:
        """Whether the newest attempt (for the causality id) provably ended.

        Consumed by stall recovery gates: a bare ``Task.done()`` proves
        nothing about the underlying HTTP thread; only a completed worker
        future (or an attempt that observed its terminal result) counts.
        """
        return self._attempts.proven_quiesced(causality_id or None)

    async def _best_effort_abort(self, session_id: str) -> None:
        """Ask OpenCode to abort work after asynchronous cancellation.

        kodmial/aa#431: the recovery abort is recovery machinery, never the
        abandoned turn itself. It must not inherit the turn-causality stamp
        — otherwise its own in-flight attempt row would mask the stalled
        turn's newest-attempt evidence at the quiescence gate.
        """
        if not session_id:
            return
        _causality_clear()
        try:
            await self.abort(session_id)
        except Exception:
            logger.info("opencode best-effort abort failed")

    @staticmethod
    def _session_id_from_path(path: str) -> str:
        """Extract a session id from a v1 or v2 session route (best effort)."""
        parts = path.split("/")
        if len(parts) >= 4 and parts[1] == "api" and parts[2] == "session" and parts[3]:
            return urllib.parse.unquote(parts[3].split("?")[0])
        if len(parts) >= 3 and parts[1] == "session" and parts[2]:
            return urllib.parse.unquote(parts[2].split("?")[0])
        if len(parts) >= 5 and parts[1] == "api" and parts[2] == "experimental":
            if parts[3] == "session" and parts[4]:
                return urllib.parse.unquote(parts[4].split("?")[0])
        return ""

    async def _run_attempt(
        self,
        *,
        operation: str,
        work: Callable[[], object],
        timeout: float | None,
        agent: str = "",
        audit_agent: str = "",
        model: str = "",
        fallback_model: str = "",
        abort_session_id: str = "",
    ) -> object:
        """Run one blocking logical attempt and publish honest evidence.

        The timeout is resolved once and passed as an immutable argument to
        the worker thread. Concurrent callers therefore never overwrite each
        other's bound. There is intentionally no ``asyncio.wait_for`` around
        the whole response: a healthy slow model is not a timeout merely
        because it exceeded an arbitrary constant.

        Attempt observability (kodmial/aa#431): the ``attempt-start`` audit
        row is written before the worker thread is spawned, every terminal
        outcome is written honestly, and a late worker completion after a
        cancelled await is published as late evidence on the same attempt —
        never as progress for a different turn.
        """
        effective_timeout = self._request_timeout if timeout is None else timeout
        if effective_timeout is not None and effective_timeout <= 0:
            raise ValueError("timeout must be > 0")
        started = time.perf_counter()
        attempt_id, row = self._open_request_row(
            operation=operation,
            agent=agent,
            audit_agent=audit_agent,
            model=model,
            fallback_model=fallback_model,
        )
        loop = asyncio.get_running_loop()
        inner: asyncio.Future[object] = loop.run_in_executor(
            None,
            functools.partial(
                self._worker_body,
                attempt_id,
                work,
                started,
            ),
        )

        def _observe_worker_outcome(done_future: asyncio.Future[object]) -> None:
            """Independent worker-observer callback (never touches loop state).

            The worker thread records its own fate directly in the ledger
            (thread-side, kodmial/aa#431); this callback only distinguishes
            the CLEAN cancellation of the executor future (the item was
            cancelled before any thread claimed it) from the asyncio-shim
            situation where the wrapping future is cancelled while the
            thread genuinely still runs. Thread-side recording is the only
            source of late outcomes in that case, so quiescence stays
            unproven until the thread itself reports a terminal state.
            """
            try:
                if done_future.cancelled():
                    # Clean cancellation before the thread ran: no orphan.
                    # The atomic ledger check also blocks a late worker claim.
                    self._attempts.mark_clean_cancel_if_not_started(attempt_id)
                    return
                # An exception completion still proves the worker future is
                # terminal. The thread-side write is authoritative, but this
                # corroborates it (and retrieves the exception so a cancelled
                # await cannot leave an unobserved-future warning).
                failure = done_future.exception()
                if failure is not None:
                    self._attempts.mark_worker_done(
                        attempt_id,
                        outcome="failed",
                        failure_category=type(failure).__name__,
                    )
            except Exception:  # noqa: BLE001 - never crash the loop on evidence
                logger.info("opencode attempt late outcome recording failed")

        # ``add_done_callback`` also runs for a future that already completed,
        # so registration immediately after submission cannot miss a fast
        # worker completion.
        inner.add_done_callback(_observe_worker_outcome)
        try:
            result = await asyncio.shield(inner)
            self._finalize_request_row(attempt_id, row, started, success=True, state="completed")
            return result
        except asyncio.CancelledError:
            # Cancellation attribution (kodmial/aa#417): a cancelled turn
            # request is bounded evidence; the underlying worker thread may
            # keep running until the socket bound or process isolation
            # applies, which the caller owns.
            inner.cancel()
            self._finalize_request_row(
                attempt_id,
                row,
                started,
                success=False,
                state="cancelled",
                failure_category="cancelled",
            )
            # The worker thread records its own real fate thread-side
            # (kodmial/aa#431): late completion/failure visibility lives in
            # _worker_body, and the executor callback above only classifies
            # a pre-thread clean cancellation. The event loop must not
            # reimplement worker evidence, and the recovery abort attempt
            # below must not mask the newest stalled-turn attempt.
            abort_task = asyncio.create_task(self._best_effort_abort(abort_session_id))
            abort_task.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )
            raise
        except TimeoutError as exc:
            self._finalize_request_row(
                attempt_id,
                row,
                started,
                success=False,
                state="failed",
                failure_category=type(exc).__name__,
            )
            raise OpenCodeTimeoutError("opencode request timed out") from exc
        except OpenCodeError as exc:
            row["error_kind"] = str(getattr(exc, "kind", ""))[:64]
            self._attempts.mark_error_kind(attempt_id, str(getattr(exc, "kind", "")))
            self._finalize_request_row(
                attempt_id,
                row,
                started,
                success=False,
                state="failed",
                failure_category=type(exc).__name__,
            )
            raise
        except OSError as exc:
            self._finalize_request_row(
                attempt_id,
                row,
                started,
                success=False,
                state="failed",
                failure_category=type(exc).__name__,
            )
            # Connection refused/reset against the loopback server: the
            # runtime is down or unreachable; retryable once it is back.
            raise OpenCodeTransientError("opencode runtime is unreachable") from exc

    async def _call(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        timeout: float | None = None,
        *,
        agent: str = "",
        audit_agent: str = "",
        model: str = "",
        fallback_model: str = "",
    ) -> object:
        """Run one HTTP request through the shared attempt supervisor."""
        effective_timeout = self._request_timeout if timeout is None else timeout
        return await self._run_attempt(
            operation=self._request_operation(method, path, body),
            work=functools.partial(self._sync_request, method, path, body, effective_timeout),
            timeout=timeout,
            agent=agent,
            audit_agent=audit_agent,
            model=model,
            fallback_model=fallback_model,
            abort_session_id=self._session_id_from_path(path),
        )

    async def health(self) -> HealthInfo:
        path = "/api/info" if self._protocol == "v2" else "/global/health"
        payload = await self._call("GET", path)
        if not isinstance(payload, dict):
            raise OpenCodeDeterministicError("opencode health returned an invalid payload")
        version = payload.get("version")
        if self._protocol == "v2":
            healthy = isinstance(version, str) and bool(version) and type(payload.get("pid")) is int
        else:
            healthy = payload.get("healthy") is True
        return HealthInfo(healthy=healthy, version=version if isinstance(version, str) else "")

    async def create_session(self, title: str = "") -> SessionInfo:
        path = "/api/session" if self._protocol == "v2" else "/session"
        payload = await self._call("POST", path, {"title": title})
        if not isinstance(payload, dict):
            raise OpenCodeDeterministicError("opencode session creation returned invalid data")
        session_id = payload.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise OpenCodeDeterministicError("opencode session creation returned no id")
        remote_title = payload.get("title")
        info = SessionInfo(
            id=session_id, title=remote_title if isinstance(remote_title, str) else ""
        )
        logger.info("opencode session created")
        return info

    async def get_session(self, session_id: str) -> SessionInfo:
        quoted = urllib.parse.quote(session_id, safe="")
        path = f"/api/session/{quoted}" if self._protocol == "v2" else f"/session/{quoted}"
        payload = await self._call("GET", path)
        if not isinstance(payload, dict):
            raise OpenCodeDeterministicError("opencode session lookup returned invalid data")
        remote_id = payload.get("id")
        if not isinstance(remote_id, str) or not remote_id:
            raise OpenCodeDeterministicError("opencode session lookup returned no id")
        remote_title = payload.get("title")
        return SessionInfo(
            id=remote_id, title=remote_title if isinstance(remote_title, str) else ""
        )

    async def delete_session(self, session_id: str) -> bool:
        quoted = urllib.parse.quote(session_id, safe="")
        path = f"/api/session/{quoted}" if self._protocol == "v2" else f"/session/{quoted}"
        await self._call("DELETE", path)
        self._v2_pinned_models.pop(session_id, None)
        logger.info("opencode session deleted")
        return True

    @property
    def served_model_audit(self) -> tuple[dict[str, str], ...]:
        """Return privacy-safe actual served-model evidence for qualification."""
        return tuple(dict(item) for item in self._served_model_audit)

    @staticmethod
    def _served_model_from_info(info: object) -> str:
        """Extract ``provider/model`` from v1 or v2 assistant metadata."""
        if not isinstance(info, dict):
            return ""
        candidates: list[object] = [info, info.get("assistant")]
        nested_model = info.get("model")
        if isinstance(nested_model, dict):
            candidates.append(nested_model)
        metadata = info.get("metadata")
        if isinstance(metadata, dict):
            candidates.extend((metadata, metadata.get("assistant")))
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            provider = candidate.get("providerID", candidate.get("provider_id", ""))
            model = candidate.get("modelID", candidate.get("model_id", candidate.get("id", "")))
            if isinstance(provider, str) and isinstance(model, str) and provider and model:
                return f"{provider}/{model}"
        return ""

    @property
    def token_usage_audit(self) -> tuple[dict[str, object], ...]:
        """Return privacy-safe model token usage keyed by logical agent."""
        return tuple(dict(item) for item in self._token_usage_audit)

    def _record_token_usage(self, info: object, *, agent: str, audit_agent: str = "") -> None:
        if not isinstance(info, dict):
            return
        tokens = info.get("tokens")
        if not isinstance(tokens, dict):
            return
        logical_agent = (audit_agent or agent or "").strip()[:128]
        entry: dict[str, object] = {"agent": logical_agent}
        found = False
        for source_key, target_key in (
            ("input", "input"),
            ("output", "output"),
            ("reasoning", "reasoning"),
        ):
            value = tokens.get(source_key)
            if isinstance(value, (int, float)) and float(value) >= 0:
                entry[target_key] = int(value)
                found = True
        cache = tokens.get("cache")
        if isinstance(cache, dict):
            for source_key, target_key in (("read", "cache_read"), ("write", "cache_write")):
                value = cache.get(source_key)
                if isinstance(value, (int, float)) and float(value) >= 0:
                    entry[target_key] = int(value)
                    found = True
        if not found:
            return
        self._token_usage_audit.append(entry)
        if len(self._token_usage_audit) > 1024:
            del self._token_usage_audit[:-1024]

    def _validate_served_model(
        self, info: object, *, requested: str, agent: str, audit_agent: str = ""
    ) -> None:
        """Fail closed when OpenCode served a model other than the pinned request."""
        expected = (requested or "").strip()
        if not expected:
            return
        served = self._served_model_from_info(info)
        if not served:
            raise OpenCodeDeterministicError("opencode response omitted served model identity")
        if served != expected:
            raise OpenCodeDeterministicError(
                f"opencode served model mismatch: expected {expected}, got {served}"
            )
        logical_agent = (audit_agent or agent or "").strip()[:128]
        self._served_model_audit.append(
            {
                "agent": logical_agent,
                "requested": expected[:128],
                "served": served[:128],
            }
        )
        if len(self._served_model_audit) > 512:
            del self._served_model_audit[:-512]

    def _v2_message_page(
        self,
        session_id: str,
        *,
        limit: int,
        order: str,
        cursor: str | None,
        timeout: float | None,
    ) -> tuple[list[dict[str, object]], str | None]:
        """Fetch one raw v2 history page and return its entries/cursor."""
        quoted = urllib.parse.quote(session_id, safe="")
        params: dict[str, str] = {"limit": str(max(1, limit))}
        if cursor:
            # The v2 contract forbids combining an opaque cursor with order.
            params["cursor"] = cursor
        else:
            params["order"] = order
        path = f"/api/session/{quoted}/message?{urllib.parse.urlencode(params)}"
        payload = self._sync_request("GET", path, None, timeout)
        return self._parse_v2_message_page(payload)

    @staticmethod
    def _parse_v2_message_page(
        payload: object,
    ) -> tuple[list[dict[str, object]], str | None]:
        if not isinstance(payload, dict):
            raise OpenCodeDeterministicError("opencode history returned invalid data")
        data = payload.get("data")
        if not isinstance(data, list):
            raise OpenCodeDeterministicError("opencode history returned invalid data")
        cursor_info = payload.get("cursor")
        next_cursor = cursor_info.get("next") if isinstance(cursor_info, dict) else None
        entries = [dict(entry) for entry in data if isinstance(entry, dict)]
        return entries, next_cursor if isinstance(next_cursor, str) and next_cursor else None

    def _v2_history_entries(
        self,
        session_id: str,
        *,
        deadline: float,
        order: str,
        page_limit: int,
        stop_id: str,
        max_pages: int = 64,
    ) -> tuple[list[dict[str, object]], bool]:
        """Walk v2 history pages (newest/oldest first) with a bounded guard.

        The walk looks for the admitted user message by its authoritative id.
        ``boundary_found`` is returned so the caller can fail closed instead
        of treating an older turn's terminal messages as this turn's
        completion when the admitted prompt has not been projected yet (or
        the bounded page walk never reaches it). Creation time alone cannot
        prove the boundary: older turns are also "older than admission".
        """
        entries: list[dict[str, object]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        boundary_found = False
        for _ in range(max_pages):
            # Every page socket bound is derived from the one shared deadline.
            # A fixed per-page timeout would let a slow server multiply the
            # caller's budget by the page limit.
            page, next_cursor = self._v2_message_page(
                session_id,
                limit=page_limit,
                order=order,
                cursor=cursor,
                timeout=_v2_remaining(deadline),
            )
            entries.extend(page)
            if stop_id and any(entry.get("id") == stop_id for entry in page):
                boundary_found = True
                break
            if not next_cursor or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return entries, boundary_found

    @staticmethod
    def _v2_created_ms(entry: object) -> int:
        """Read a v2 message's ``time.created`` epoch milliseconds, if any."""
        if not isinstance(entry, dict):
            return 0
        timestamp = entry.get("time")
        if not isinstance(timestamp, dict):
            return 0
        created = timestamp.get("created")
        if isinstance(created, bool) or not isinstance(created, (int, float)):
            return 0
        return int(created)

    def _v2_enqueue_prompt(
        self, session_id: str, text: str, timeout: float | None
    ) -> tuple[str, int]:
        """Enqueue one v2 prompt and return its admitted id and creation time.

        A 200 response is durable admission evidence only. It says nothing
        about whether an assistant reply exists yet, so both the id and the
        admission timestamp are carried forward to correlate the eventual
        assistant message with *this* prompt.
        """
        quoted = urllib.parse.quote(session_id, safe="")
        payload = self._sync_request(
            "POST", f"/api/session/{quoted}/prompt", {"text": text}, timeout
        )
        if not isinstance(payload, dict):
            raise OpenCodeDeterministicError("opencode prompt enqueue returned invalid data")
        message_id = payload.get("id")
        message_type = payload.get("type")
        if not isinstance(message_id, str) or not message_id or message_type != "user":
            raise OpenCodeDeterministicError("opencode prompt enqueue returned no admitted message")
        return message_id, self._v2_created_ms(payload)

    def _v2_wait_for_idle(self, session_id: str, timeout: float | None) -> None:
        quoted = urllib.parse.quote(session_id, safe="")
        self._sync_request("POST", f"/api/experimental/session/{quoted}/wait", None, timeout)

    def _v2_turn_messages(
        self,
        session_id: str,
        *,
        user_message_id: str,
        user_created_ms: int,
        deadline: float,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        """Return ``(newest_assistant, idle)`` belonging to this turn only.

        Entries are newest-first. The walk stops at *this* turn's admitted
        user message, so a previous turn's assistant/idle message that
        happens to share the final history page can never be reported as
        this turn's result. Both the admitted id and the admission
        timestamp are enforced: the id is authoritative, while the
        timestamp still rejects strictly older messages if the server ever
        rewrites the id.

        If the admitted user message has not appeared in history yet (or the
        bounded page walk never reaches it), no candidate belongs to this
        turn. Returning ``(None, None)`` keeps the caller polling until its
        deadline rather than reusing a previous turn's completed reply.
        """
        entries, boundary_found = self._v2_history_entries(
            session_id,
            deadline=deadline,
            order="desc",
            page_limit=100,
            stop_id=user_message_id,
        )
        if not boundary_found:
            return None, None
        assistant: dict[str, object] | None = None
        idle: dict[str, object] | None = None
        for entry in entries:
            # Both boundaries are enforced: the admitted id is authoritative,
            # and creation time rejects anything strictly older than this
            # turn even if the server rewrote the id. Equal-millisecond
            # messages stay eligible because a fast assistant reply can share
            # the user message's timestamp.
            if user_message_id and entry.get("id") == user_message_id:
                break
            created_ms = self._v2_created_ms(entry)
            if user_created_ms and 0 < created_ms < user_created_ms:
                break
            entry_type = entry.get("type")
            if entry_type == "assistant" and assistant is None:
                assistant = entry
            elif entry_type == "idle" and idle is None:
                idle = entry
        return assistant, idle

    def _v2_completed_assistant(
        self,
        session_id: str,
        *,
        user_message_id: str,
        user_created_ms: int,
        timeout: float | None,
    ) -> dict[str, object]:
        """Await and return the assistant message of a genuinely completed turn.

        A v2 prompt response is durable admission, not a reply. Completion is
        proven only by the terminal ``idle`` message created *after* this
        turn's user message; that message carries the authoritative outcome
        for the turn. The loop re-issues ``/wait`` because the agent loop may
        not have started when the prompt was admitted, and it stops at a
        caller-supplied deadline (or a finite fallback) so a queued prompt can
        never be reported as a finished response.

        A succeeded turn must also expose an assistant message; a failed or
        interrupted turn is classified from the structured error carried by
        its idle message. Waiting, queued, missing, or failed results are
        never converted into a fictitious reply.
        """
        budget = _V2_COMPLETION_FALLBACK_TIMEOUT_SECONDS if timeout is None else timeout
        deadline = time.monotonic() + max(0.001, budget)
        while True:
            # Recompute the remaining budget before every blocking call. The
            # wait route and each history page must share the caller's one
            # deadline, otherwise a slow-but-successful call could let a
            # single poll overshoot it by a multiple of the budget.
            self._v2_wait_for_idle(session_id, _v2_remaining(deadline))
            assistant, idle = self._v2_turn_messages(
                session_id,
                user_message_id=user_message_id,
                user_created_ms=user_created_ms,
                deadline=deadline,
            )
            if idle is not None:
                outcome = idle.get("outcome")
                error = idle.get("error")
                if isinstance(error, dict) and error:
                    # A structured failure is authoritative even if the
                    # outcome field claims success: never turn a provider
                    # error into a PASS.
                    raise classify_structured_error(error)
                if outcome != "succeeded":
                    if isinstance(outcome, str) and outcome:
                        raise OpenCodeDeterministicError(
                            f"opencode session did not complete successfully: outcome={outcome}"
                        )
                    raise OpenCodeDeterministicError(
                        "opencode session did not report a completion outcome"
                    )
                if assistant is None:
                    raise OpenCodeDeterministicError(
                        "opencode completion produced no assistant reply"
                    )
                error = assistant.get("error")
                if isinstance(error, dict) and error:
                    raise classify_structured_error(error)
                finish = assistant.get("finish")
                if isinstance(finish, str) and finish and finish != "stop":
                    raise OpenCodeDeterministicError(
                        f"opencode assistant reply did not finish normally: {finish}"
                    )
                return assistant
            if time.monotonic() >= deadline:
                raise OpenCodeTimeoutError(
                    "opencode turn completion was not observed before the deadline"
                )
            # Never sleep past the deadline: the next iteration raises instead
            # of idling beyond the caller's bound.
            time.sleep(
                min(
                    _V2_COMPLETION_POLL_SECONDS,
                    max(0.0, deadline - time.monotonic()),
                )
            )

    def _v2_text_exchange(
        self,
        session_id: str,
        text: str,
        system: str,
        timeout: float | None,
        agent: str,
        model: str,
        audit_agent: str,
    ) -> str:
        prompt = _v2_system_preamble(system, text)
        # One deadline covers admission plus completion, so a slow enqueue
        # cannot extend the caller's bound.
        deadline = _v2_budget_deadline(timeout)
        user_message_id, user_created_ms = self._v2_enqueue_prompt(
            session_id, prompt, _v2_remaining(deadline)
        )
        assistant = self._v2_completed_assistant(
            session_id,
            user_message_id=user_message_id,
            user_created_ms=user_created_ms,
            timeout=_v2_remaining(deadline),
        )
        self._validate_served_model(
            assistant, requested=model, agent=agent, audit_agent=audit_agent
        )
        self._record_token_usage(assistant, agent=agent, audit_agent=audit_agent)
        return _extract_v2_content(assistant)

    def _v2_structured_exchange(
        self,
        session_id: str,
        text: str,
        system: str,
        schema: dict[str, object],
        retry_count: int,
        timeout: float | None,
        agent: str,
        model: str,
        audit_agent: str,
    ) -> dict[str, object]:
        prompt = _v2_structured_prompt(text, system=system, schema=schema)
        # The caller's bound covers the whole structured exchange, including
        # every JSON retry; a retry may not multiply the wall-clock budget.
        deadline = _v2_budget_deadline(timeout)
        for attempt in range(retry_count + 1):
            current_prompt = prompt
            if attempt:
                current_prompt = (
                    f"{prompt}\n\nYour previous reply was not one valid JSON object. "
                    "Return only the JSON object now."
                )
            user_message_id, user_created_ms = self._v2_enqueue_prompt(
                session_id, current_prompt, _v2_remaining(deadline)
            )
            assistant = self._v2_completed_assistant(
                session_id,
                user_message_id=user_message_id,
                user_created_ms=user_created_ms,
                timeout=_v2_remaining(deadline),
            )
            self._validate_served_model(
                assistant, requested=model, agent=agent, audit_agent=audit_agent
            )
            self._record_token_usage(assistant, agent=agent, audit_agent=audit_agent)
            parsed = _parse_v2_json_object(_extract_v2_content(assistant))
            if parsed is not None:
                return parsed
        raise OpenCodeDeterministicError("opencode structured output missing")

    async def _v2_configure_selectors(
        self,
        session_id: str,
        *,
        timeout: float | None,
        agent: str,
        model: str,
        audit_agent: str,
    ) -> None:
        """Pin the session model/agent before a v2 prompt turn.

        A model pin is session-scoped and persists on the server. An empty
        model hint means "server default applies", which this client cannot
        restore through the documented v2 surface (the model route requires a
        concrete ``Model.Ref``). Refuse the omission after a pin so the turn
        cannot be served by a stale model pointer instead of the default.
        """
        quoted = urllib.parse.quote(session_id, safe="")
        model_ref = model_ref_payload(model)
        if model_ref is not None:
            await self._call(
                "POST",
                f"/api/session/{quoted}/model",
                {"model": model_ref},
                timeout=timeout,
                agent=agent,
                audit_agent=audit_agent,
                model=model,
            )
            self._v2_pinned_models[session_id] = model
        elif self._v2_pinned_models.get(session_id):
            raise OpenCodeDeterministicError(
                "opencode session already has a pinned model; refusing to omit "
                "the model selector after a previous pin"
            )
        if agent:
            await self._call(
                "POST",
                f"/api/session/{quoted}/agent",
                {"agent": agent},
                timeout=timeout,
                agent=agent,
                audit_agent=audit_agent,
                model=model,
            )

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
        system: str = "",
        format: dict[str, object] | None = None,
        audit_agent: str = "",
    ) -> str:
        if not text or not text.strip():
            raise OpenCodeDeterministicError("refusing to send an empty prompt")
        if self._protocol == "v2":
            if format is not None:
                raise OpenCodeDeterministicError(
                    "OpenCode v2 send_message has no native format support; "
                    "use send_structured_message"
                )
            if timeout is not None and timeout <= 0:
                raise ValueError("timeout must be > 0")
            await self._v2_configure_selectors(
                session_id,
                timeout=timeout,
                agent=agent,
                model=model,
                audit_agent=audit_agent,
            )
            effective_timeout = self._request_timeout if timeout is None else timeout
            result = await self._run_attempt(
                operation="message-text",
                work=functools.partial(
                    self._v2_text_exchange,
                    session_id,
                    text,
                    system,
                    effective_timeout,
                    agent,
                    model,
                    audit_agent,
                ),
                timeout=timeout,
                agent=agent,
                audit_agent=audit_agent,
                model=model,
                abort_session_id=session_id,
            )
            logger.info("opencode message completed")
            if not isinstance(result, str):
                raise OpenCodeDeterministicError("opencode message returned no text")
            return result
        quoted = urllib.parse.quote(session_id, safe="")
        body: dict[str, object] = {"parts": [{"type": "text", "text": text}]}
        if agent:
            body["agent"] = agent
        if system.strip():
            body["system"] = system
        if format is not None:
            body["format"] = format
        parsed_model = model_payload(model)
        if parsed_model is not None:
            body["model"] = parsed_model
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be > 0")
        payload = await self._call(
            "POST",
            f"/session/{quoted}/message",
            body,
            timeout=timeout,
            agent=agent,
            audit_agent=audit_agent,
            model=model,
        )
        if not isinstance(payload, dict):
            raise OpenCodeDeterministicError("opencode prompt returned invalid data")
        info = payload.get("info")
        if isinstance(info, dict) and info.get("error") not in (None, False):
            raise classify_provider_error(info.get("error"))
        self._validate_served_model(info, requested=model, agent=agent, audit_agent=audit_agent)
        self._record_token_usage(info, agent=agent, audit_agent=audit_agent)
        reply = _extract_text(payload.get("parts"))
        # Never log the prompt or the reply; only the fact of completion.
        logger.info("opencode message completed")
        return reply

    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
        system: str = "",
        schema: dict[str, object],
        retry_count: int = 2,
        audit_agent: str = "",
    ) -> dict[str, object]:
        """Send one structured request and return its validated object.

        The v1 wire uses OpenCode's native ``json_schema`` format. The v2
        prompt route has no format field, so v2 sends an explicit JSON-only
        schema instruction (and retries it) before parsing strictly.
        """
        if not text or not text.strip():
            raise OpenCodeDeterministicError("refusing to send an empty prompt")
        if retry_count < 0:
            raise ValueError("retry_count must be >= 0")
        if self._protocol == "v2":
            if timeout is not None and timeout <= 0:
                raise ValueError("timeout must be > 0")
            await self._v2_configure_selectors(
                session_id,
                timeout=timeout,
                agent=agent,
                model=model,
                audit_agent=audit_agent,
            )
            effective_timeout = self._request_timeout if timeout is None else timeout
            result = await self._run_attempt(
                operation="message-structured",
                work=functools.partial(
                    self._v2_structured_exchange,
                    session_id,
                    text,
                    system,
                    schema,
                    retry_count,
                    effective_timeout,
                    agent,
                    model,
                    audit_agent,
                ),
                timeout=timeout,
                agent=agent,
                audit_agent=audit_agent,
                model=model,
                abort_session_id=session_id,
            )
            if not isinstance(result, dict):
                raise OpenCodeDeterministicError("opencode structured output missing")
            logger.info("opencode structured message completed")
            return result
        quoted = urllib.parse.quote(session_id, safe="")
        body: dict[str, object] = {
            "parts": [{"type": "text", "text": text}],
            "format": {"type": "json_schema", "schema": schema, "retryCount": retry_count},
        }
        if agent:
            body["agent"] = agent
        if system.strip():
            body["system"] = system
        parsed_model = model_payload(model)
        if parsed_model is not None:
            body["model"] = parsed_model
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be > 0")
        payload = await self._call(
            "POST",
            f"/session/{quoted}/message",
            body,
            timeout=timeout,
            agent=agent,
            audit_agent=audit_agent,
            model=model,
        )
        if not isinstance(payload, dict):
            raise OpenCodeDeterministicError("opencode structured output returned invalid data")
        info = payload.get("info")
        if isinstance(info, dict) and info.get("error") not in (None, False):
            raise classify_provider_error(info.get("error"))
        self._validate_served_model(info, requested=model, agent=agent, audit_agent=audit_agent)
        self._record_token_usage(info, agent=agent, audit_agent=audit_agent)
        structured = _extract_structured(payload)
        if structured is None:
            raise OpenCodeDeterministicError("opencode structured output missing")
        logger.info("opencode structured message completed")
        return structured

    async def list_messages(self, session_id: str, *, limit: int = 50) -> list[ChatMessage]:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if limit == 0:
            return []
        if self._protocol == "v2":
            collected: list[ChatMessage] = []
            cursor: str | None = None
            seen_cursors: set[str] = set()
            while len(collected) < limit:
                page_limit = min(max(limit - len(collected), 1), 100)
                params: dict[str, str] = {"limit": str(page_limit)}
                if cursor:
                    params["cursor"] = cursor
                else:
                    params["order"] = "desc"
                quoted = urllib.parse.quote(session_id, safe="")
                path = f"/api/session/{quoted}/message?{urllib.parse.urlencode(params)}"
                payload = await self._call("GET", path)
                page, next_cursor = self._parse_v2_message_page(payload)
                for entry in page:
                    entry_type = entry.get("type")
                    if entry_type not in ("user", "assistant"):
                        continue
                    text = _extract_v2_content(entry)
                    if entry_type == "user":
                        raw_text = entry.get("text")
                        text = raw_text if isinstance(raw_text, str) else ""
                    collected.append(ChatMessage(role=entry_type, text=text))
                    if len(collected) >= limit:
                        break
                if not next_cursor or next_cursor in seen_cursors:
                    break
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            collected.reverse()
            return collected[:limit]
        quoted = urllib.parse.quote(session_id, safe="")
        payload = await self._call("GET", f"/session/{quoted}/message?limit={limit}")
        if not isinstance(payload, list):
            raise OpenCodeDeterministicError("opencode history returned invalid data")
        messages: list[ChatMessage] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            info = entry.get("info")
            role = ""
            if isinstance(info, dict):
                raw_role = info.get("role")
                role = raw_role if isinstance(raw_role, str) else ""
            messages.append(ChatMessage(role=role, text=_extract_text(entry.get("parts"))))
        return messages

    async def abort(self, session_id: str) -> bool:
        """Cancel in-flight server work via the native interrupt route.

        v1 uses ``POST /session/{id}/abort`` (204). v2 uses
        ``POST /api/session/{id}/interrupt`` and reports whether an active
        execution owned by this server was actually interrupted.
        """
        quoted = urllib.parse.quote(session_id, safe="")
        if self._protocol == "v2":
            payload = await self._call("POST", f"/api/session/{quoted}/interrupt")
            if not isinstance(payload, dict):
                return False
            return payload.get("interrupted") is True
        await self._call("POST", f"/session/{quoted}/abort")
        return True


@dataclass
class FakeOpenCodeClient(OpenCodeClient):
    """In-memory fake of the OpenCode boundary for unit tests.

    Mirrors the real session semantics (opaque ids, create/continue/delete,
    per-session histories) without processes or network. Failures matching
    the real taxonomy can be injected via :attr:`fail_next`.
    """

    latency: float = 0.0
    fail_next: OpenCodeError | None = None
    _sessions: dict[str, dict[str, object]] = field(default_factory=dict, init=False)
    _counter: int = field(default=0, init=False)

    async def _settle(self, timeout: float | None = None) -> None:
        if self.fail_next is not None:
            failure = self.fail_next
            self.fail_next = None
            raise failure
        delay = self.latency
        if timeout is not None and delay > timeout:
            await asyncio.sleep(timeout)
            raise OpenCodeTimeoutError("opencode request timed out")
        if delay > 0:
            await asyncio.sleep(delay)

    async def health(self) -> HealthInfo:
        await self._settle()
        return HealthInfo(healthy=True, version="fake")

    async def create_session(self, title: str = "") -> SessionInfo:
        await self._settle()
        self._counter += 1
        session_id = f"ses_fake{self._counter:06d}"
        self._sessions[session_id] = {"title": title, "messages": []}
        return SessionInfo(id=session_id, title=title)

    async def get_session(self, session_id: str) -> SessionInfo:
        await self._settle()
        record = self._sessions.get(session_id)
        if record is None:
            raise OpenCodeSessionNotFoundError("opencode session not found: fake...")
        title = record.get("title")
        return SessionInfo(id=session_id, title=title if isinstance(title, str) else "")

    async def delete_session(self, session_id: str) -> bool:
        await self._settle()
        if session_id not in self._sessions:
            raise OpenCodeSessionNotFoundError("opencode session not found: fake...")
        del self._sessions[session_id]
        return True

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
        system: str = "",
        format: dict[str, object] | None = None,
        audit_agent: str = "",
    ) -> str:
        if not text or not text.strip():
            raise OpenCodeDeterministicError("refusing to send an empty prompt")
        # Validate the model hint exactly like the real client.
        model_payload(model)
        _ = (system, format, audit_agent)
        await self._settle(timeout)
        record = self._sessions.get(session_id)
        if record is None:
            raise OpenCodeSessionNotFoundError("opencode session not found: fake...")
        history = record["messages"]
        assert isinstance(history, list)
        history.append({"role": "user"})
        reply = f"Фиктивный ответ {len(history)}"
        history.append({"role": "assistant", "text": reply})
        return reply

    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
        system: str = "",
        schema: dict[str, object],
        retry_count: int = 2,
        audit_agent: str = "",
    ) -> dict[str, object]:
        """Fake native structured output: returns the queued object or empty plan."""
        if not text or not text.strip():
            raise OpenCodeDeterministicError("refusing to send an empty prompt")
        model_payload(model)
        _ = (system, schema, retry_count, audit_agent)
        await self._settle(timeout)
        record = self._sessions.get(session_id)
        if record is None:
            raise OpenCodeSessionNotFoundError("opencode session not found: fake...")
        history = record["messages"]
        assert isinstance(history, list)
        history.append({"role": "user"})
        queued = getattr(self, "structured_queue", None)
        if isinstance(queued, list) and queued:
            reply_obj = queued.pop(0)
            if not isinstance(reply_obj, dict):
                raise OpenCodeDeterministicError("fake structured queue must hold dicts")
            reply: dict[str, object] = dict(reply_obj)
        else:
            reply = {"queries": []}
        history.append({"role": "assistant", "text": "", "structured": reply})
        return reply

    async def list_messages(self, session_id: str, *, limit: int = 50) -> list[ChatMessage]:
        await self._settle()
        record = self._sessions.get(session_id)
        if record is None:
            raise OpenCodeSessionNotFoundError("opencode session not found: fake...")
        history = record.get("messages")
        assert isinstance(history, list)
        selected = history[-limit:] if limit < len(history) else history
        messages: list[ChatMessage] = []
        for entry in selected:
            assert isinstance(entry, dict)
            role = entry.get("role")
            text = entry.get("text")
            messages.append(
                ChatMessage(
                    role=role if isinstance(role, str) else "",
                    text=text if isinstance(text, str) else "",
                )
            )
        return messages

    async def abort(self, session_id: str) -> bool:
        await self._settle()
        if session_id not in self._sessions:
            raise OpenCodeSessionNotFoundError("opencode session not found: fake...")
        return True
