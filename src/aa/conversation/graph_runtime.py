"""LangGraph AA runtime for the Telegram production boundary (issue #118).

This module is the only ordinary conversational path after the cutover::

    Telegram -> safety/commands -> typing heartbeat -> LangGraph thread
      -> planner/retrieval/evidence/AA Agent/grounding -> Telegram delivery

Design:

- one Telegram private chat maps deterministically to one LangGraph
  ``thread_id`` via :func:`thread_id_for_chat` (one-way digest; the raw
  chat identifier is never logged or stored);
- the graph/checkpointer is the authoritative conversation-memory layer;
  accumulated OpenCode session history is never a second memory (hidden
  planner/summarizer/verifier/final-generation calls use the thin #113
  :class:`OpenCodeChatModel` adapter with fresh ephemeral OpenCode
  sessions created, invoked and deleted per call);
- one local ``opencode serve`` process per worker/runtime is kept (the
  runtime is owned by :class:`Application`, never duplicated here);
- no TUI/Electron, no second LLM/provider client;
- ``/new`` clears only that chat's LangGraph thread state via the
  checkpointer (no cross-chat leakage);
- text and voice share one conversation state: voice transcripts enter
  this exact boundary as normal user turns after local ASR.

The runtime is transport-independent so the production-boundary suite can
exercise the exact boundary Telegram uses. A lightweight in-memory
delegate mode exists for fast unit tests; production uses the compiled
LangGraph turn graph with the RAM-resident retrieval index.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("aa.conversation.graph_runtime")

TurnDelegate = Callable[[str, str], Awaitable[str]]
"""Test delegate: ``(thread_id, user_message) -> reply``."""


class GraphRuntimeError(ValueError):
    """Deterministic runtime failure (typed unsuccessful turn outcome)."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"graph runtime failed [{category}]" + (f": {detail}" if detail else ""))
        self.category = category
        self.detail = detail


class GraphTurnRuntime:
    """Owns the LangGraph turn graph and per-chat thread lifecycle.

    Production mode binds a compiled graph plus a live checkpointer;
    delegate mode (tests) runs an injected async function with an
    in-memory per-thread history so continuity/isolation semantics can be
    proven without models or a database.
    """

    def __init__(
        self,
        *,
        delegate: TurnDelegate | None = None,
        checkpointer_factory: Any | None = None,
        graph: Any | None = None,
    ) -> None:
        self._delegate = delegate
        self._factory = checkpointer_factory
        self._graph = graph
        self._checkpointer: Any | None = None
        self._checkpointer_ctx: Any | None = None
        self._histories: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        self._running = False
        # Privacy-safe per-turn stage telemetry (counts/latencies/outcomes
        # only). Keyed by thread digest; never holds user or evidence text.
        self._last_telemetry: dict[str, dict[str, Any]] = {}
        # Last certified candidate/certificate per thread (digests + outcome
        # only for telemetry; full candidate text stays in graph state).
        # Delivery receipts distinguish provisional-certified graph memory
        # from acknowledged Telegram acceptance (#304 seam, #312 owns
        # quote-history mutation).
        self._last_certificates: dict[str, dict[str, Any]] = {}
        self._last_receipts: dict[str, list[dict[str, Any]]] = {}

    @property
    def running(self) -> bool:
        """Whether the runtime is started."""
        return self._running

    @property
    def compiled_graph(self) -> Any | None:
        """The bound compiled graph (``None`` in delegate mode)."""
        return self._graph

    def thread_id(self, chat_id: int) -> str:
        """Map one Telegram chat deterministically to one thread id."""
        from aa.conversation.memory import thread_id_for_chat

        return thread_id_for_chat(chat_id)

    async def start(self) -> None:
        """Enter the checkpointer context and mark the runtime running."""
        if self._running:
            return
        if self._delegate is not None:
            self._running = True
            return
        if self._factory is not None and self._graph is None:
            ctx = self._factory.checkpointer()
            saver = await ctx.__aenter__()
            self._checkpointer_ctx = ctx
            self._checkpointer = saver
            self._graph = self._build_graph(saver)
        self._running = True

    async def stop(self) -> None:
        """Exit the checkpointer context (idempotent)."""
        self._running = False
        ctx, self._checkpointer_ctx = self._checkpointer_ctx, None
        self._checkpointer = None
        if self._factory is not None:
            self._graph = None
        if ctx is not None:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                logger.info("graph runtime checkpointer close failed")

    def attach_graph(self, graph: Any, *, checkpointer: Any | None = None) -> None:
        """Bind an already-compiled graph (tests/tooling)."""
        self._graph = graph
        if checkpointer is not None:
            self._checkpointer = checkpointer

    def _build_graph(self, saver: Any) -> Any:
        """Compile the production turn graph lazily (deferred imports)."""
        raise GraphRuntimeError("no-graph-factory", "production graph factory is not bound")

    async def run_turn(self, chat_id: int, text: str) -> str:
        """Run one ordinary user turn in that chat's thread and return text."""
        import time as _time

        cleaned = text.strip() if isinstance(text, str) else ""
        if not cleaned:
            raise GraphRuntimeError("empty-turn", "refusing an empty turn")
        thread = self.thread_id(chat_id)
        if self._delegate is not None:
            async with self._lock:
                history = self._histories.setdefault(thread, [])
            started = _time.perf_counter()
            try:
                reply = await self._delegate(thread, cleaned)
            except GraphRuntimeError:
                raise
            except Exception as exc:
                # Provider 429 retires the runner for fresh-VM checkpoint
                # recovery; it must never collapse into a GraphRuntimeError
                # (which the application would serve as a natural fallback
                # and lose the resume signal). No blind retry here: the
                # caller owns retire/restart.
                from aa.opencode.errors import OpenCodeRateLimitError as _DelegateRateLimit

                if isinstance(exc, _DelegateRateLimit):
                    raise
                raise GraphRuntimeError("delegate-failed", type(exc).__name__) from exc
            if not isinstance(reply, str) or not reply.strip():
                raise GraphRuntimeError("empty-reply", "delegate returned no text")
            # Delegate/test seam certification (#304): even the in-memory
            # delegate produces a serializable candidate/certificate so the
            # application boundary enforces the same digest gate. Delegate
            # turns carry no book evidence (clarification kind) and still
            # pass safety/envelope/presentation certification.
            try:
                from aa.conversation.finalization import AnswerCandidate as _DCandidate
                from aa.conversation.finalization import (
                    certify_candidate as _DCertify,
                )
                from aa.conversation.finalization import (
                    context_digest_for_turn as _DContext,
                )
                from aa.conversation.finalization import (
                    normalize_answer_text as _DNormalize,
                )

                normalized_reply = _DNormalize(reply)
                delegate_context = _DContext(
                    question=cleaned,
                    resolved_intent=cleaned,
                    summary="",
                    recent_texts=[],
                )
                delegate_candidate = _DCandidate(
                    text=normalized_reply,
                    evidence_bundle=[],
                    context_digest=delegate_context,
                    outcome_kind="clarification",
                )
                try:
                    delegate_certificate = _DCertify(
                        candidate=delegate_candidate,
                        grounding_result=None,
                        question=cleaned,
                        resolved_intent=cleaned,
                        summary="",
                        recent_messages=[],
                    )
                except Exception as exc:
                    # Fail closed: an uncertifiable delegate reply is
                    # never delivered, appended to history, or masked
                    # with a synthesized negative certificate.
                    raise GraphRuntimeError("uncertified-reply", type(exc).__name__) from exc
                try:
                    from aa.conversation.finalization import (
                        verify_certificate as _DVerify,
                    )

                    _DVerify(
                        candidate=delegate_candidate,
                        certificate=delegate_certificate,
                    )
                except GraphRuntimeError:
                    raise
                except Exception as exc:
                    raise GraphRuntimeError("uncertified-reply", type(exc).__name__) from exc
                try:
                    self._last_certificates[thread] = {
                        "candidate": delegate_candidate.model_dump(mode="json"),
                        "certificate": delegate_certificate.model_dump(mode="json"),
                    }
                except Exception:
                    pass
            except GraphRuntimeError:
                raise
            except Exception as exc:
                raise GraphRuntimeError("uncertified-reply", type(exc).__name__) from exc
            async with self._lock:
                history.append(cleaned)
                history.append(normalized_reply)
            elapsed_ms = (_time.perf_counter() - started) * 1000.0
            logger.info(
                "graph turn completed",
                extra={"reply_len": len(normalized_reply), "latency_ms": round(elapsed_ms, 1)},
            )
            return normalized_reply
        graph = self._graph
        if graph is None:
            raise GraphRuntimeError("not-started", "graph runtime has no compiled graph")
        # Never attribute a previous turn's grounding telemetry to a later
        # turn that crashes or is converted to an application fallback.
        self._last_telemetry.pop(thread, None)
        self._last_certificates.pop(thread, None)
        started = _time.perf_counter()
        try:
            result = await self._invoke_graph(graph, thread, cleaned)
        except GraphRuntimeError:
            raise
        except Exception as exc:
            # Provider 429 (planner, retrieval selection, answer, verifier
            # or whole-turn judge) retires the runner for fresh-VM
            # checkpoint recovery; wrapping it as "graph-failed" would let
            # the application serve a natural fallback and lose the resume
            # signal. Propagate without blind retry: the caller owns
            # retire/restart and the checkpointer owns resume.
            from aa.opencode.errors import OpenCodeRateLimitError as _TurnRateLimit

            if isinstance(exc, _TurnRateLimit):
                raise
            raise GraphRuntimeError("graph-failed", type(exc).__name__) from exc
        elapsed_ms = (_time.perf_counter() - started) * 1000.0
        final_raw = result.get("final_response", "") or result.get("draft_response", "")
        if not isinstance(final_raw, str) or not final_raw.strip():
            raise GraphRuntimeError("empty-reply", "graph returned no text")
        # Single delivery gate enforcement (kodmial/aa#304): the compiled
        # graph must certify the exact normalized text against the exact
        # evidence/context digests before send. Stale or missing
        # certificates reject delivery fail-closed and never enter history
        # as qualified answers. No .strip()/clipping/substitution after
        # certification: the certified text travels byte-for-byte.
        try:
            from aa.conversation.finalization import (
                AnswerCandidate as _Candidate,
            )
            from aa.conversation.finalization import (
                VerificationCertificate as _Certificate,
            )
            from aa.conversation.finalization import (
                normalize_answer_text as _normalize,
            )
            from aa.conversation.finalization import (
                verify_certificate as _verify_cert,
            )

            candidate_raw = result.get("answer_candidate", None)
            certificate_raw = result.get("verification_certificate", None)
            if str(result.get("route", "normal")) != "normal":
                # Control route (command/blocked/emergency): finalize
                # returns no book candidate/certificate by design, so the
                # book-certification gate must not reject it here. Control
                # text still requires a non-empty reply and never consumes
                # a book certificate.
                if not isinstance(final_raw, str) or not final_raw.strip():
                    raise GraphRuntimeError("empty-reply", "graph returned no text")
                final = final_raw
                try:
                    self._last_certificates.pop(thread, None)
                except Exception:
                    pass
                self._record_stage_telemetry(thread, result, elapsed_ms, len(final))
                logger.info(
                    "graph turn completed",
                    extra={
                        "reply_len": len(final),
                        "latency_ms": round(elapsed_ms, 1),
                        "route": str(result.get("route", "normal")),
                    },
                )
                return final
            if not isinstance(candidate_raw, dict) or not isinstance(certificate_raw, dict):
                raise GraphRuntimeError("uncertified-reply", "missing delivery certificate")
            candidate = _Candidate.model_validate(candidate_raw)
            certificate = _Certificate.model_validate(certificate_raw)
            pack_raw = result.get("evidence_pack", [])
            pack = (
                [dict(item) for item in pack_raw if isinstance(item, dict)]
                if isinstance(pack_raw, list)
                else []
            )
            _verify_cert(candidate=candidate, certificate=certificate, evidence_pack=pack)
            # The graph's final_response must be the certified text exactly;
            # a borrowed verdict for mutated text fails closed here.
            if _normalize(final_raw) != candidate.text or final_raw != candidate.text:
                raise GraphRuntimeError(
                    "stale-certificate", "final text mutated after certification"
                )
            final = candidate.text
            try:
                stored_texts = result.get("response_unit_texts", [])
                self._last_certificates[thread] = {
                    "candidate": candidate.model_dump(mode="json"),
                    "certificate": certificate.model_dump(mode="json"),
                    "response_unit_texts": [
                        dict(item) for item in stored_texts if isinstance(item, dict)
                    ]
                    if isinstance(stored_texts, list)
                    else [],
                }
            except Exception:
                pass
        except GraphRuntimeError:
            raise
        except Exception as exc:
            raise GraphRuntimeError("uncertified-reply", type(exc).__name__) from exc
        self._record_stage_telemetry(thread, result, elapsed_ms, len(final))
        logger.info(
            "graph turn completed",
            extra={
                "reply_len": len(final),
                "latency_ms": round(elapsed_ms, 1),
                "route": str(result.get("route", "normal")),
            },
        )
        return final

    def _record_stage_telemetry(
        self, thread: str, result: dict[str, Any], total_ms: float, reply_len: int
    ) -> None:
        """Store one privacy-safe stage snapshot for later qualification."""
        try:
            retry = result.get("retry_state", {})
            retry_d = dict(retry) if isinstance(retry, dict) else {}
            embedded = retry_d.get("turn_telemetry", {})
            embedded_d = dict(embedded) if isinstance(embedded, dict) else {}
            queries = result.get("search_queries", [])
            query_count = len(queries) if isinstance(queries, list) else 0
            pack = result.get("evidence_pack", [])
            pack_count = len(pack) if isinstance(pack, list) else 0
            grounding = result.get("grounding_result", {})
            grounding_d = dict(grounding) if isinstance(grounding, dict) else {}
            grounding_units = grounding_d.get("units", [])
            response_units = len(grounding_units) if isinstance(grounding_units, list) else 0
            try:
                from aa.conversation.answer_adequacy import planner_reason_for as _tele_reason
                from aa.conversation.answer_adequacy import runtime_sha as _tele_sha
                from aa.conversation.answer_adequacy import trace_id_for_snapshot as _tele_trace
            except Exception:
                _tele_reason = None  # type: ignore[assignment]
                _tele_sha = None  # type: ignore[assignment]
                _tele_trace = None  # type: ignore[assignment]
            _planner_outcome = str(
                embedded_d.get("planner_outcome", "invoked" if query_count else "glue")
            )
            _planner_reason = str(embedded_d.get("planner_reason", "") or "")
            if not _planner_reason and _tele_reason is not None:
                try:
                    _planner_reason = str(
                        _tele_reason(
                            int(embedded_d.get("planner_query_count", query_count) or 0),
                            _planner_outcome,
                        )
                    )
                except Exception:
                    _planner_reason = "unknown"
            _trace_id = str(embedded_d.get("turn_trace_id", "") or "")
            if not _trace_id and _tele_trace is not None:
                try:
                    _trace_id = str(_tele_trace(dict(embedded_d)))
                except Exception:
                    _trace_id = ""
            _runtime_sha = str(embedded_d.get("runtime_sha", "") or "")
            if not _runtime_sha and _tele_sha is not None:
                try:
                    _runtime_sha = str(_tele_sha())
                except Exception:
                    _runtime_sha = "unknown"
            snapshot: dict[str, Any] = {
                "planner_query_count": int(embedded_d.get("planner_query_count", query_count)),
                "planner_outcome": _planner_outcome,
                "planner_reason": _planner_reason or "unknown",
                "planner_latency_ms": float(embedded_d.get("planner_latency_ms", 0.0)),
                "retrieval_passages": int(embedded_d.get("retrieval_passages", pack_count)),
                "retrieval_hits": int(
                    embedded_d.get(
                        "retrieval_hits", embedded_d.get("retrieval_passages", pack_count)
                    )
                ),
                "retrieval_outcome": str(embedded_d.get("retrieval_outcome", "unknown")),
                "retrieval_latency_ms": float(
                    embedded_d.get(
                        "retrieval_latency_ms",
                        float(result.get("retrieval_latency_ms", 0.0) or 0.0),
                    )
                ),
                "retrieval_over_budget": bool(result.get("retrieval_over_budget", False)),
                "answer_outcome": str(embedded_d.get("answer_outcome", "unknown")),
                "answer_latency_ms": float(embedded_d.get("answer_latency_ms", 0.0)),
                "answer_rounds": int(embedded_d.get("answer_rounds", 0)),
                "verifier_outcome": str(embedded_d.get("verifier_outcome", "unknown")),
                "verifier_latency_ms": float(embedded_d.get("verifier_latency_ms", 0.0)),
                "verifier_unavailable_units": int(embedded_d.get("verifier_unavailable_units", 0)),
                "response_units": int(response_units),
                # Count only book-scoped, verifier-supported units with
                # actual passage provenance. No user or corpus text leaks.
                "verified_book_units": sum(
                    1
                    for unit in grounding_units
                    if isinstance(unit, dict)
                    and unit.get("scope") == "book"
                    and unit.get("supported") is True
                    and bool(unit.get("evidence_passage_ids"))
                ),
                "repair_rounds": int(embedded_d.get("repair_rounds", 0)),
                "repair_budget_exceeded": bool(embedded_d.get("repair_budget_exceeded", False)),
                "turn_budget_exceeded": bool(embedded_d.get("turn_budget_exceeded", False)),
                "all_required_supported": bool(grounding_d.get("all_required_supported", False)),
                "adequacy_verdict": str(embedded_d.get("adequacy_verdict", "unknown")),
                "failure_category": str(embedded_d.get("failure_category", "")),
                "answers_request": bool(embedded_d.get("answers_request", False)),
                "technically_grounded": bool(embedded_d.get("technically_grounded", False)),
                "qualified": bool(embedded_d.get("qualified", False)),
                "turn_trace_id": _trace_id,
                "runtime_sha": _runtime_sha or "unknown",
                "total_latency_ms": round(total_ms, 1),
                "reply_len": int(reply_len),
                # Truthful selection route (#312): provisional graph
                # telemetry never claims delivery. ``last_telemetry`` is
                # not a delivery acknowledgment; confirmed delivery lives
                # only in transport receipts.
                "selection_route": str(
                    embedded_d.get("selection_route", embedded_d.get("pack_order", "unknown"))
                    or "unknown"
                ),
                "semantic_selection_applied": bool(
                    embedded_d.get("semantic_selection_applied", False)
                ),
                "pack_order": str(embedded_d.get("pack_order", "unknown") or "unknown"),
                "coverage_status": str(embedded_d.get("coverage_status", "unknown") or "unknown"),
                "pack_digest": str(embedded_d.get("pack_digest", "") or ""),
                "evidence_discovered": int(embedded_d.get("evidence_discovered", 0) or 0),
                "evidence_previewed": int(embedded_d.get("evidence_previewed", 0) or 0),
                "evidence_read": int(embedded_d.get("evidence_read", 0) or 0),
                "evidence_selected_in_pack": int(
                    embedded_d.get("evidence_selected_in_pack", pack_count) or 0
                ),
                "evidence_used_in_answer": int(embedded_d.get("evidence_used_in_answer", 0) or 0),
                "evidence_quoted_to_user": int(embedded_d.get("evidence_quoted_to_user", 0) or 0),
                "evidence_delivered": "pending-delivery",
                "delivery_outcome": "provisional-certified",
                "delivery_status": str(result.get("delivery_status", "provisional-certified")),
            }
            self._last_telemetry[thread] = snapshot
            logger.info(
                "v2 turn telemetry",
                extra={
                    "planner_outcome": snapshot["planner_outcome"],
                    "planner_reason": snapshot["planner_reason"],
                    "retrieval_outcome": snapshot["retrieval_outcome"],
                    "answer_outcome": snapshot["answer_outcome"],
                    "answer_latency_ms": snapshot["answer_latency_ms"],
                    "verifier_outcome": snapshot["verifier_outcome"],
                    "verifier_latency_ms": snapshot["verifier_latency_ms"],
                    "verifier_unavailable_units": snapshot["verifier_unavailable_units"],
                    "response_units": snapshot["response_units"],
                    "adequacy_verdict": snapshot["adequacy_verdict"],
                    "failure_category": snapshot["failure_category"],
                    "repair_budget_exceeded": snapshot["repair_budget_exceeded"],
                    "latency_ms": snapshot["total_latency_ms"],
                },
            )
        except Exception:
            pass

    def last_telemetry_for_thread(self, thread: str) -> dict[str, Any]:
        """Return the last privacy-safe stage snapshot for ``thread``."""
        return dict(self._last_telemetry.get(thread, {}))

    def last_certificate_for_thread(self, thread: str) -> dict[str, Any]:
        """Return the last certified candidate/certificate for ``thread``."""
        return dict(self._last_certificates.get(thread, {}))

    def last_receipts_for_thread(self, thread: str) -> list[dict[str, Any]]:
        """Return the last transport receipts for ``thread`` (provisional vs ack)."""
        return [dict(item) for item in self._last_receipts.get(thread, [])]

    def record_delivery_receipts(self, thread: str, receipts: list[dict[str, Any]]) -> None:
        """Persist transport acknowledgments without mutating certified text.

        Graph/checkpointer memory stays provisional-certified until Telegram
        confirms; receipts record confirmed/failed/unknown per segment so a
        partially delivered reply is never presented as complete history.
        """
        try:
            self._last_receipts[thread] = [dict(item) for item in (receipts or [])]
        except Exception:
            pass

    async def _invoke_graph(self, graph: Any, thread: str, text: str) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage

        config = {"configurable": {"thread_id": thread}}
        payload: dict[str, Any] = {
            "messages": [HumanMessage(content=text)],
            "current_user_message": text,
        }
        result = await graph.ainvoke(payload, config=config)
        if not isinstance(result, dict):
            raise GraphRuntimeError("graph-failed", "graph returned no state")
        return dict(result)

    def commit_confirmed_delivery(self, thread: str) -> list[dict[str, Any]]:
        """Commit receipt-confirmed book quotes (no text, idempotent).

        Reads the last certified candidate/certificate plus transport
        receipts for ``thread`` and returns the delivered quote ranges
        to persist. Never derives delivery from graph success or
        telemetry alone: empty receipts or missing certificates commit
        nothing. Failed/unknown segments yield only a
        ``possibly-delivered`` safety record, never confirmed history.
        """
        try:
            cert_store = self._last_certificates.get(thread, {})
            receipts = self._last_receipts.get(thread, [])
            if not isinstance(cert_store, dict) or not receipts:
                return []
            candidate = cert_store.get("candidate", {})
            certificate = cert_store.get("certificate", {})
            if not isinstance(candidate, dict) or not isinstance(certificate, dict):
                return []
            candidate_text = str(candidate.get("text", "") or "")
            evidence_bundle = candidate.get("evidence_bundle", [])
            pack = (
                [dict(item) for item in evidence_bundle if isinstance(item, dict)]
                if isinstance(evidence_bundle, list)
                else []
            )
            raw_verdicts = certificate.get("claim_verdicts", [])
            verdicts = (
                [dict(item) for item in raw_verdicts if isinstance(item, dict)]
                if isinstance(raw_verdicts, list)
                else []
            )
            stored_texts: list[dict[str, Any]] = []
            try:
                maybe_texts = cert_store.get("response_unit_texts", [])
                if isinstance(maybe_texts, list):
                    stored_texts = [dict(item) for item in maybe_texts if isinstance(item, dict)]
            except Exception:
                stored_texts = []
            certificate_id = str(certificate.get("certificate_id", "") or "")
            from aa.conversation.quote_state import delivered_ranges_from_candidate

            confirmed, possibly = delivered_ranges_from_candidate(
                certified_text=candidate_text,
                evidence_pack=pack,
                claim_verdicts=verdicts,
                stored_unit_texts=stored_texts or None,
                receipts=[dict(item) for item in receipts if isinstance(item, dict)],
                certificate_id=certificate_id,
            )
            return [*confirmed, *possibly]
        except Exception:
            return []

    async def apply_delivery_commit(self, thread: str) -> list[dict[str, Any]]:
        """Persist receipt-confirmed quotes into checkpointer state.

        Computes :meth:`commit_confirmed_delivery` and merges it into
        the thread's ``recent_quote_ranges`` via ``aupdate_state`` when
        a compiled graph is bound. Idempotent: replays deduplicate and
        session reset clears history. Returns the committed ranges.
        """
        from aa.conversation.quote_state import merge_recent_ranges

        committed = self.commit_confirmed_delivery(thread)
        if not committed:
            return []
        graph = self._graph
        if graph is None:
            return committed
        try:
            config = {"configurable": {"thread_id": thread}}
            try:
                snapshot = await graph.aget_state(config)
                current_raw = snapshot.values.get("recent_quote_ranges", [])
            except Exception:
                try:
                    snapshot = graph.get_state(config)
                    current_raw = snapshot.values.get("recent_quote_ranges", [])
                except Exception:
                    current_raw = []
            current = [dict(item) for item in (current_raw or []) if isinstance(item, dict)]
            merged = merge_recent_ranges(current, committed)
            try:
                await graph.aupdate_state(config, {"recent_quote_ranges": merged})
            except Exception:
                try:
                    graph.update_state(config, {"recent_quote_ranges": merged})
                except Exception:
                    pass
            return committed
        except Exception:
            return committed

    async def clear_chat(self, chat_id: int) -> None:
        """Clear only that chat's LangGraph conversation state (``/new``)."""
        thread = self.thread_id(chat_id)
        if self._delegate is not None:
            async with self._lock:
                self._histories.pop(thread, None)
            try:
                self._last_telemetry.pop(thread, None)
                self._last_certificates.pop(thread, None)
                self._last_receipts.pop(thread, None)
            except Exception:
                pass
            logger.info("graph thread cleared")
            return
        saver = self._checkpointer
        if saver is None:
            return
        try:
            deleter = getattr(saver, "adelete_thread", None)
            if callable(deleter):
                await deleter(thread)
            else:
                sync = getattr(saver, "delete_thread", None)
                if callable(sync):
                    await asyncio.to_thread(sync, thread)
            logger.info("graph thread cleared")
        except Exception:
            logger.info("graph thread clear failed")

    def history_for_thread(self, thread: str) -> list[str]:
        """Return the delegate-mode history for ``thread`` (tests only)."""
        return list(self._histories.get(thread, []))


def build_production_runtime(
    *,
    client: Any,
    settings: Any,
    index: Any | None,
    checkpoint_dir: Path | None = None,
) -> GraphTurnRuntime:
    """Build the production runtime bound to OpenCode models + index.

    Hidden planner/summarizer/verifier/answer calls use the thin #113
    model adapter (ephemeral OpenCode sessions per call, never polluting
    the user conversation). The caller owns ``start()``/``stop()`` around
    the checkpointer context. The compiled graph is built at ``start()``
    time inside :class:`_ProductionGraphRuntime`.
    """
    return _ProductionGraphRuntime(client=client, settings=settings, index=index)


class _ProductionGraphRuntime(GraphTurnRuntime):
    """Production specialization compiling the real turn graph on start."""

    def __init__(self, *, client: Any, settings: Any, index: Any | None) -> None:
        super().__init__()
        self._client = client
        self._settings = settings
        self._index = index

    def _build_graph(self, saver: Any) -> Any:  # pragma: no cover - production wiring
        from aa.conversation.graph import build_turn_graph
        from aa.conversation.memory import default_memory_config
        from aa.conversation.model_adapter import (
            ANSWER_AGENT_V2,
            SUMMARIZER_AGENT_V2,
            build_planner_model,
        )
        from aa.retrieval.evidence import RetrievalConfig

        primary = str(getattr(self._settings, "opencode_model", ""))
        fallback = str(getattr(self._settings, "opencode_fallback_model", ""))
        planner = build_planner_model(
            self._client,
            primary_model=primary,
            fallback_model=fallback,
        )
        summarizer = planner.with_agent(SUMMARIZER_AGENT_V2)
        answer = planner.with_agent(ANSWER_AGENT_V2)
        # Product invariant (kodmial/aa#202): the grounding verifier uses
        # Muse Spark only with no fallback. Space Bunny is deliberately
        # excluded from this critical gate. The logical audit identity
        # stays ``aa-verifier-v2`` while the transport agent selector is
        # omitted (server default applies) so a custom-agent rejection can
        # never strand the Muse verifier; the verifier system prompt still
        # travels through the native ``system`` field. A regression test
        # locks this routing; other agents retain the configured
        # primary/fallback policy. Never rebuild the verifier via
        # ``planner.with_agent(...)``: that would inherit the fallback.
        from aa.config import DEFAULT_PRIMARY_MODEL
        from aa.conversation.model_adapter import build_verifier_model

        verifier = build_verifier_model(
            self._client,
            primary_model=DEFAULT_PRIMARY_MODEL,
            request_timeout=planner.request_timeout,
        )
        return build_turn_graph(
            planner_model=planner,
            summary_model=summarizer,
            memory_config=default_memory_config(),
            checkpointer=saver,
            retrieval_index=self._index,
            retrieval_config=RetrievalConfig(),
            answer_model=answer,
            verifier_model=verifier,
        )

    async def start(self) -> None:
        if self.running:
            return
        from aa.conversation.memory import SqliteCheckpointerFactory, default_memory_config

        factory = SqliteCheckpointerFactory(default_memory_config())
        self._factory = factory
        await super().start()
        # super().start() calls _build_graph(saver) via the factory hook.
        # Keep the factory for cleanup visibility.
        self._factory = factory

    async def stop(self) -> None:
        await super().stop()
        factory = self._factory
        self._factory = None
        if factory is not None:
            cleanup = getattr(factory, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    logger.info("graph runtime checkpoint cleanup failed")


__all__ = [
    "GraphRuntimeError",
    "GraphTurnRuntime",
    "TurnDelegate",
    "build_production_runtime",
]
