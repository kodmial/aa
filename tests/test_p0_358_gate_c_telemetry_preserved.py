"""P0 kodmial/aa#358: Gate C typed-failure telemetry must survive the fallback.

Proven product failure on exact main 26762c3 (run 38225484508):

- C:live-book-grounding-substantive-drinking-2 on live-production-path
  (plus D:workflow-step-failed downstream and E:latency-budget-exceeded
  SLO).

Live diagnostics for the failing run show the functional defect is in
the production fallback path, not only latency: 7 turns collapsed to
the marked service-error text (live-answer-no-generic-collapse), while
5 regression scenarios (including the blocking drinking-2) delivered
text with an entirely empty stage snapshot (planner_outcome "",
production_outcome "", verified_book_units 0, no stage latencies) and
the independent judge reported 6 missing-turn-telemetry unavailable
cases. The pipeline therefore served a fallback with no attributable
stage telemetry, so Gate C could only report a generic grounding
failure instead of the concrete broken stage.

Root cause: GraphTurnRuntime.run_turn pops the thread telemetry at turn
start and records a new snapshot only on the certified-success path.
When the graph raises a typed TurnFailed (answer/verifier/repair
budget paths carry a full privacy-safe telemetry dict) or when the
delivery certification gate rejects (missing/stale certificate,
empty reply), the exception path wrapped everything as
GraphRuntimeError("graph-failed") without preserving the available
stage snapshot. The delivered service-error text then carried no
telemetry for qualification.

Repair (generic, no exact-question special cases; Product Contract
#110 unchanged): preserve the typed-failure telemetry as this thread's
last snapshot before raising the typed GraphRuntimeError, and preserve
the graph result's retry_state.turn_telemetry on certification-gate
failures. The turn still fails as a typed unsuccessful outcome (the
application still serves the marked service error, never synthetic
conversation); only the privacy-safe attribution survives.
"""

from __future__ import annotations

from typing import Any

from aa.conversation.failures import TurnFailed
from aa.conversation.graph_runtime import GraphRuntimeError, GraphTurnRuntime


def _failed_telemetry() -> dict[str, Any]:
    return {
        "planner_query_count": 12,
        "planner_outcome": "ok",
        "planner_reason": "substantive-with-queries",
        "planner_latency_ms": 120.0,
        "retrieval_passages": 3,
        "retrieval_outcome": "evidence-ready",
        "retrieval_latency_ms": 250.0,
        "answer_outcome": "failed",
        "answer_latency_ms": 300.0,
        "verifier_outcome": "skipped-turn-budget",
        "verifier_latency_ms": 0.0,
        "verifier_unavailable_units": 0,
        "response_units": 0,
        "verified_book_units": 0,
        "adequacy_verdict": "unknown",
        "failure_category": "verifier-turn-budget",
        "answers_request": True,
        "technically_grounded": False,
        "qualified": False,
    }


async def test_turn_failed_telemetry_survives_graph_failed() -> None:
    runtime = GraphTurnRuntime()
    telemetry = _failed_telemetry()

    async def _raise_turn_failed(graph: Any, thread: str, text: str) -> dict[str, Any]:
        _ = (graph, thread, text)
        raise TurnFailed(
            "verifier-turn-budget", "verifier round skipped", telemetry=dict(telemetry)
        )

    runtime.attach_graph(object())
    runtime._invoke_graph = _raise_turn_failed  # type: ignore[method-assign]
    await runtime.start()
    try:
        try:
            await runtime.run_turn(923581, "generic substantive request")
        except GraphRuntimeError as exc:
            assert exc.category == "graph-failed"
        else:
            raise AssertionError("typed failure must still raise GraphRuntimeError")
        snapshot = runtime.last_telemetry_for_thread(runtime.thread_id(923581))
        assert snapshot, "failed-turn telemetry must not be empty"
        assert snapshot.get("failure_category") == "verifier-turn-budget"
        assert snapshot.get("planner_outcome") == "ok"
        assert snapshot.get("retrieval_outcome") == "evidence-ready"
        assert snapshot.get("verifier_outcome") == "skipped-turn-budget"
    finally:
        await runtime.stop()


async def test_chained_turn_failed_telemetry_survives() -> None:
    runtime = GraphTurnRuntime()
    telemetry = _failed_telemetry()

    async def _raise_wrapped(graph: Any, thread: str, text: str) -> dict[str, Any]:
        _ = (graph, thread, text)
        try:
            raise TurnFailed("answer-failed", "no draft", telemetry=dict(telemetry))
        except TurnFailed as inner:
            raise RuntimeError("langgraph node wrapper") from inner

    runtime.attach_graph(object())
    runtime._invoke_graph = _raise_wrapped  # type: ignore[method-assign]
    await runtime.start()
    try:
        try:
            await runtime.run_turn(923582, "generic substantive request")
        except GraphRuntimeError:
            pass
        else:
            raise AssertionError("wrapped failure must still raise GraphRuntimeError")
        snapshot = runtime.last_telemetry_for_thread(runtime.thread_id(923582))
        assert snapshot.get("failure_category") == "verifier-turn-budget"
    finally:
        await runtime.stop()


async def test_certification_failure_preserves_result_telemetry() -> None:
    from aa.conversation.finalization import (
        AnswerCandidate,
        VerificationCertificate,
        WholeAnswerVerdict,
        context_digest_for_turn,
        normalize_answer_text,
    )

    runtime = GraphTurnRuntime()
    runtime.attach_graph(object())
    await runtime.start()
    try:
        question = "generic substantive request"
        resolved = "generic resolved intent"
        digest = context_digest_for_turn(
            question=question, resolved_intent=resolved, summary="", recent_texts=[]
        )
        candidate = AnswerCandidate(
            text=normalize_answer_text("Поддержка рядом помогает пережить тягу сегодня."),
            evidence_bundle=[],
            context_digest=digest,
            outcome_kind="clarification",
        )
        certificate = VerificationCertificate(
            certificate_id="cert-358",
            answer_sha256="0" * 64,
            evidence_digest="e" * 16,
            context_digest=digest,
            claim_verdicts=[],
            whole_answer_verdict=WholeAnswerVerdict(
                supported=True,
                coverage_ok=True,
                conditions_preserved=True,
            ),
        )
        result: dict[str, Any] = {
            "final_response": "mutated text that breaks the certificate",
            "draft_response": "mutated text that breaks the certificate",
            "answer_candidate": candidate.model_dump(mode="json"),
            "verification_certificate": certificate.model_dump(mode="json"),
            "evidence_pack": [],
            "route": "normal",
            "retry_state": {"turn_telemetry": dict(_failed_telemetry())},
        }

        async def _stale_result(graph: Any, thread: str, text: str) -> dict[str, Any]:
            _ = (graph, thread, text)
            return dict(result)

        runtime._invoke_graph = _stale_result  # type: ignore[method-assign]
        try:
            await runtime.run_turn(923583, question)
        except GraphRuntimeError as exc:
            assert exc.category in ("stale-certificate", "uncertified-reply", "graph-failed")
        else:
            raise AssertionError("stale certificate must fail closed")
        snapshot = runtime.last_telemetry_for_thread(runtime.thread_id(923583))
        assert snapshot.get("failure_category") == "verifier-turn-budget"
    finally:
        await runtime.stop()


def test_no_exact_question_special_cases() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    source = (root / "src" / "aa" / "conversation" / "graph_runtime.py").read_text(encoding="utf-8")
    for fragment in (
        "тянет выпить",
        "тянеет выпить",
        "Поругались дома",
        "поругались дома",
        "не могу успокоиться",
        "мысли крутятся",
        "покупать акции",
        "одному не получается",
        "чем помочь можешь",
    ):
        assert fragment not in source
