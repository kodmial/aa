"""Delivery vs product accounting on the latency evidence (kodmial/aa#436).

History (Gate C run 39343286141): latency evidence was counted while the
very same turn was delivered as a marked service error and no telemetry
survived for most failed stages, so Gate C reported only a generic
collapse and Gate E kept serving PASS. The contract pinned here:

- a delivered service error keeps its latency observation AND fails the
  reliability accounting (never aggregated as a served success);
- a delivery timeout keeps its censored latency and fails reliability;
- legacy samples without the new fields degrade to explicit ``unknown``
  (never auto-success, never an invented reliability failure);
- typed ``answer_failure_cause`` survives the graph failure into the
  delivery stamp (``product_outcome``/``failure_reason``) without losing
  the earliest cause.
"""

from __future__ import annotations

from typing import Any

import pytest

from aa.qualification.gate_latency import (
    C_SERIES,
    RELIABILITY_FAILED,
    TurnSample,
    build_c_series_from_live_summary,
    evaluate_latency_report,
    samples_from_legacy_latencies,
    validate_samples,
)

CLOCK = "monotonic-perf_counter"


# ---------------------------------------------------------------------------
# TurnSample semantics.
# ---------------------------------------------------------------------------


def test_delivered_service_error_is_product_failure() -> None:
    sample = TurnSample(
        causality_id="c1",
        family="ordinary-text",
        mode="text",
        outcome="served",
        latency_ms=100.0,
        delivery_status="delivered",
        product_outcome="service_error",
        failure_reason="provider-rate-limit",
    )
    assert sample.product_failure
    assert not sample.product_success


def test_legacy_sample_is_inconclusive_never_auto_success_or_failure() -> None:
    sample = TurnSample(
        causality_id="c2",
        family="ordinary-text",
        mode="text",
        outcome="served",
        latency_ms=100.0,
    )
    assert sample.delivery_status == "unknown"
    assert sample.product_outcome == "unknown"
    assert not sample.product_failure
    assert not sample.product_success


def test_timeout_sample_counts_as_product_failure() -> None:
    sample = TurnSample(
        causality_id="c3",
        family="ordinary-text",
        mode="text",
        outcome="timeout",
        latency_ms=120000.0,
        censored=True,
        delivery_status="not_attempted",
        product_outcome="timeout",
    )
    assert sample.censored
    assert sample.product_failure


# ---------------------------------------------------------------------------
# Report evaluation.
# ---------------------------------------------------------------------------


def _sample(cid: str, **overrides: Any) -> TurnSample:
    base: dict[str, Any] = dict(
        causality_id=cid,
        family="ordinary-text",
        mode="text",
        outcome="served",
        latency_ms=100.0,
    )
    base.update(overrides)
    return TurnSample(**base)


def test_delivered_service_error_fails_reliability_with_observed_latency() -> None:
    report = evaluate_latency_report(
        [
            _sample("c1", delivery_status="delivered", product_outcome="answer_success"),
            _sample(
                "c2",
                latency_ms=150.0,
                delivery_status="delivered",
                product_outcome="service_error",
                failure_reason="verifier-unavailable",
            ),
        ],
        mode="observe",
    )
    assert report.status == "FAIL"
    assert report.failure_category == RELIABILITY_FAILED
    assert report.product_failure_count == 1
    assert report.product_success_count == 1
    assert report.max_ms == pytest.approx(150.0)


def test_censored_timeout_fails_reliability() -> None:
    report = evaluate_latency_report(
        [
            _sample(
                "c1",
                outcome="timeout",
                latency_ms=120000.0,
                censored=True,
                delivery_status="not_attempted",
                product_outcome="timeout",
            ),
        ],
        mode="observe",
    )
    assert report.status == "FAIL"
    assert report.failure_category == RELIABILITY_FAILED


def test_legacy_unknown_product_sample_does_not_fail_reliability() -> None:
    report = evaluate_latency_report(
        [
            _sample("c1", latency_ms=100.0),
            _sample("c2", latency_ms=120.0),
        ],
        mode="observe",
    )
    # All product outcomes unknown: inconclusive, never auto-failure.
    assert report.product_failure_count == 0
    if report.status == "PASS":
        assert report.product_success_count == 0


def test_empty_report_has_zero_product_counts_and_stays_blocked() -> None:
    report = evaluate_latency_report([], mode="observe")
    assert report.status == "BLOCKED"
    assert report.product_failure_count == 0
    assert report.product_success_count == 0


# ---------------------------------------------------------------------------
# Legacy sources.
# ---------------------------------------------------------------------------


def test_samples_from_legacy_latencies_stay_unknown() -> None:
    samples = samples_from_legacy_latencies([100.0, 120.0])
    validate_samples(samples)
    assert all(sample.product_outcome == "unknown" for sample in samples)
    assert all(sample.delivery_status == "unknown" for sample in samples)


def test_build_c_series_typed_samples() -> None:
    summary: dict[str, Any] = {
        "lanes": [
            {
                "lane": "live-telegram-evidence",
                "metrics": {
                    "turn_latency_samples": [
                        {
                            "causality_id": "c1",
                            "family": "ordinary-text",
                            "mode": "text",
                            "outcome": "served",
                            "latency_ms": 100.0,
                            "monotonic": True,
                            "clock": CLOCK,
                            "series": C_SERIES,
                            "boundary": "production-update-intake-to-simulated-handoff",
                            "receipt_kind": "simulated-in-process",
                            "censored": False,
                            "delivery_status": "delivered",
                            "product_outcome": "service_error",
                            "failure_reason": "verifier-unavailable",
                        },
                        {
                            "causality_id": "c2",
                            "family": "ordinary-text",
                            "mode": "text",
                            "outcome": "timeout",
                            "latency_ms": 120000.0,
                            "monotonic": True,
                            "clock": CLOCK,
                            "series": C_SERIES,
                            "boundary": "production-update-intake-to-simulated-handoff",
                            "receipt_kind": "simulated-in-process",
                            "censored": True,
                            "delivery_status": "not_attempted",
                            "product_outcome": "timeout",
                            "failure_reason": "no-real-progress",
                        },
                    ],
                },
            }
        ],
    }
    samples = build_c_series_from_live_summary(summary)
    assert len(samples) == 2
    assert samples[0].product_failure
    assert samples[1].product_failure
    assert samples[1].failure_reason == "no-real-progress"
    # The suite result keeps observed timings and fails reliability.
    report = evaluate_latency_report(samples, mode="observe")
    assert report.status == "FAIL"
    assert report.failure_category == RELIABILITY_FAILED
    assert report.served_count == 1
    assert report.censored_count == 1


# ---------------------------------------------------------------------------
# Application boundary: typed cause survives to the delivery stamp.
# ---------------------------------------------------------------------------


def _sample_payload_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "causality_id": "cafedeadbeef",
        "family": "ordinary-text",
        "mode": "text",
        "outcome": "served",
        "latency_ms": 100.0,
        "monotonic": True,
        "clock": CLOCK,
        "series": C_SERIES,
        "boundary": "production-update-intake-to-simulated-handoff",
        "receipt_kind": "simulated-in-process",
        "censored": False,
        "delivery_status": "delivered",
        "product_outcome": str(snapshot.get("product_outcome", "")),
        "failure_reason": str(snapshot.get("failure_reason", "")),
    }


async def test_delivered_service_error_keeps_typed_cause_and_fails() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.answer_causality import (
        AnswerFailureCause,
        failure_reason_from_telemetry,
    )
    from aa.conversation.failures import SERVICE_ERROR_REPLY, TurnFailed
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.telegram.transport import StubTelegramTransport

    telemetry: dict[str, Any] = {
        "planner_outcome": "ok",
        "retrieval_outcome": "evidence-ready",
        "answer_outcome": "failed",
        "verifier_outcome": "unavailable",
        "verifier_latency_ms": 4.2,
        "failure_category": "clarification-unavailable",
        "answer_failure_cause": AnswerFailureCause.bounded(
            "verifier-unavailable", "verification", "verification"
        ).to_dict(),
    }

    async def _raise_turn_failed(graph: Any, thread: str, text: str) -> dict[str, Any]:
        _ = (graph, thread, text)
        raise TurnFailed("clarification-unavailable", "typed collapse", telemetry=dict(telemetry))

    settings = Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.02"})
    runtime = GraphTurnRuntime()
    runtime.attach_graph(object())
    runtime._invoke_graph = _raise_turn_failed  # type: ignore[method-assign]
    app = Application(
        settings,
        transport=StubTelegramTransport(),
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        reply = await app.respond(923201, "тяга вечером, что делать?")
        assert reply == SERVICE_ERROR_REPLY
        thread = runtime.thread_id(923201)
        snapshot = runtime.last_telemetry_for_thread(thread)
        # The typed cause travelled through the graph failure intact.
        assert snapshot["answer_failure_cause"]["reason"] == "verifier-unavailable"
        # The delivery stamp separates delivery from product outcome.
        assert snapshot["product_outcome"] == "service_error"
        assert snapshot["failure_reason"] == failure_reason_from_telemetry(snapshot)
        assert snapshot["failure_reason"] == "verifier-unavailable"
        # Harness-shaped sample fails the Gate E report honestly.
        samples = TurnSample.from_dict(_sample_payload_from_snapshot(snapshot))
        report = evaluate_latency_report([samples], mode="observe")
        assert report.status == "FAIL"
        assert report.failure_category == RELIABILITY_FAILED
        assert report.product_failure_count == 1
    finally:
        await app.stop()


async def test_healthy_delegate_turn_is_never_invented_as_product_failure() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.conversation.turn_pipeline import contains_cyrillic, leaks_internal_terms
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.telegram.transport import StubTelegramTransport

    async def _delegate(thread: str, text: str) -> str:
        _ = thread
        return "Про тягу: поддержка рядом помогает пережить вечер. Что сейчас важнее?"

    settings = Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.02"})
    runtime = GraphTurnRuntime(delegate=_delegate)
    app = Application(
        settings,
        transport=StubTelegramTransport(),
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        reply = await app.respond(923202, "тяга вечером, что делать?")
        assert contains_cyrillic(reply)
        assert not leaks_internal_terms(reply)
        thread = runtime.thread_id(923202)
        snapshot = runtime.last_telemetry_for_thread(thread)
        # Delegate-seam telemetry may be entirely absent (no snapshot);
        # there is then nothing to invent: the product outcome stays
        # inconclusive, never diagnosed as a failure or a success.
        assert snapshot.get("product_outcome", "unknown") == "unknown"
        # Such a sample does not fail reliability on pure ignorance.
        sample = TurnSample.from_dict(
            {**_sample_payload_from_snapshot(snapshot), "delivery_status": "delivered"}
        )
        assert not sample.product_failure
        assert not sample.product_success
    finally:
        await app.stop()
