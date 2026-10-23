"""Local OpenCode invocation boundary.

The narrowest stable boundary the local ``opencode serve`` runtime supports
is its HTTP API over loopback (observed on OpenCode 1.18.34):

- ``GET /global/health`` for readiness;
- ``POST /session`` / ``GET /session/{id}`` / ``DELETE /session/{id}`` for
  session lifecycle;
- ``POST /session/{id}/message`` for prompting with
  ``{"parts": [{"type": "text", "text": ...}]}``;
- ``GET /session/{id}/message`` for history;
- ``POST /session/{id}/abort`` for cancelling in-flight work.

Only the standard library is used (``urllib`` executed in a worker thread),
so the worker gains no second LLM client and no HTTP vendor dependency.
Model/provider (Zen) configuration stays on the OpenCode side: the worker
either omits the model (server default applies) or forwards the opaque
``OPENCODE_MODEL`` ``provider/model`` pointer. No credentials are handled
here and prompt text is never logged.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
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
)

logger = logging.getLogger("aa.opencode.client")


@dataclass(frozen=True)
class HealthInfo:
    """Result of the ``GET /global/health`` readiness probe."""

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
    """Translate an ``OPENCODE_MODEL`` hint into an OpenCode model payload.

    The hint is an opaque ``provider/model`` pointer owned by the OpenCode
    side. An empty hint returns ``None`` (the server default, i.e. the
    Zen-configured model, applies). A malformed non-empty hint raises a
    deterministic error so misconfiguration surfaces immediately.
    """
    if not hint:
        return None
    if "/" in hint:
        provider, _, model = hint.partition("/")
        if provider and model:
            return {"providerID": provider, "modelID": model}
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

    def __init__(self, base_url: str, *, request_timeout: float | None = None) -> None:
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
        url = f"{self._base_url}{path}"
        data: bytes | None = None
        headers = {"Accept": "application/json"}
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
        return json.loads(raw.decode("utf-8"))

    def _worker_body(
        self,
        attempt_id: str,
        method: str,
        path: str,
        body: dict[str, object] | None,
        timeout: float | None,
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
            result = self._sync_request(method, path, body, timeout)
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
        if method == "GET" and path == "/global/health":
            return "health"
        if method == "POST" and path == "/session":
            return "session-create"
        if path.startswith("/session/") and path.endswith("/message") and method == "POST":
            fmt = body.get("format") if isinstance(body, dict) else None
            return "message-structured" if isinstance(fmt, dict) else "message-text"
        if path.startswith("/session/") and path.endswith("/abort") and method == "POST":
            return "session-abort"
        if path.startswith("/session/") and method == "DELETE":
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

    async def _best_effort_abort(self, path: str) -> None:
        """Ask OpenCode to abort work after asynchronous cancellation.

        kodmial/aa#431: the recovery abort is recovery machinery, never the
        abandoned turn itself. It must not inherit the turn-causality stamp
        — otherwise its own in-flight attempt row would mask the stalled
        turn's newest-attempt evidence at the quiescence gate.
        """
        parts = path.split("/")
        if len(parts) >= 3 and parts[1] == "session" and parts[2]:
            _causality_clear()
            try:
                await self.abort(urllib.parse.unquote(parts[2]))
            except Exception:
                logger.info("opencode best-effort abort failed")

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
        """Run one HTTP request without any shared or total wall-clock race.

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
        operation = self._request_operation(method, path, body)
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
                method,
                path,
                body,
                effective_timeout,
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
            abort_task = asyncio.create_task(self._best_effort_abort(path))
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

    async def health(self) -> HealthInfo:
        payload = await self._call("GET", "/global/health")
        if not isinstance(payload, dict):
            raise OpenCodeDeterministicError("opencode health returned an invalid payload")
        healthy = payload.get("healthy")
        version = payload.get("version")
        return HealthInfo(
            healthy=healthy is True,
            version=version if isinstance(version, str) else "",
        )

    async def create_session(self, title: str = "") -> SessionInfo:
        payload = await self._call("POST", "/session", {"title": title})
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
        payload = await self._call("GET", f"/session/{quoted}")
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
        await self._call("DELETE", f"/session/{quoted}")
        logger.info("opencode session deleted")
        return True

    @property
    def served_model_audit(self) -> tuple[dict[str, str], ...]:
        """Return privacy-safe actual served-model evidence for qualification."""
        return tuple(dict(item) for item in self._served_model_audit)

    @staticmethod
    def _served_model_from_info(info: object) -> str:
        """Extract provider/model from OpenCode assistant response metadata."""
        if not isinstance(info, dict):
            return ""
        candidates: list[object] = [info, info.get("assistant")]
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
        """Send one native ``json_schema`` request and return its object."""
        if not text or not text.strip():
            raise OpenCodeDeterministicError("refusing to send an empty prompt")
        if retry_count < 0:
            raise ValueError("retry_count must be >= 0")
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
        """Cancel in-flight server work via ``POST /session/{id}/abort``.

        This is the concrete HTTP transport implementation; the abstract
        method exists only so test fakes can share the same boundary.
        """
        quoted = urllib.parse.quote(session_id, safe="")
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
