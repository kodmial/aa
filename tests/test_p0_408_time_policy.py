"""Runtime time policy and mode-aware Gate E (kodmial/aa#408)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import scripts.run_self_proving_qualification as runner  # noqa: E402
from aa.config import Settings
from aa.conversation.retrieval_node import (
    _retrieval_preserves_downstream,
    _retrieval_semantic_stage_remaining_s,
)
from aa.conversation.turn_budget import (
    TurnBudgetExpired,
    await_model_under_turn_budget,
    new_turn_budget,
)
from aa.qualification.gate_evidence import verdict_summary_v2
from aa.qualification.gate_latency import (
    C_SERIES,
    TurnSample,
    build_c_series_from_live_summary,
    evaluate_latency_report,
    new_causality_id,
)
from aa.qualification.self_proving import (
    FinalVerdict,
    GateEvidence,
    refusal_explanation,
    slo_guards,
)
from aa.time_policy import (
    OBSERVE_DETAIL,
    TimePolicy,
    TimePolicyError,
    current_time_policy,
)

SHA = "a" * 40
PRODUCT = "c" * 64
RUNTIME = "d" * 64


def _sample(latency_ms: float, *, outcome: str = "served") -> TurnSample:
    return TurnSample(
        causality_id=new_causality_id(),
        family="ordinary-text",
        mode="text",
        outcome=outcome,
        latency_ms=float(latency_ms),
        series=C_SERIES,
        censored=outcome in ("timeout", "failed"),
    )


def test_defaults_disable_runtime_limits_and_observe_gate_e(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AA_TURN_TIME_LIMITS_ENABLED",
        "AA_LATENCY_GATE_MODE",
        "AA_LATENCY_P95_TARGET_MS",
        "AA_LATENCY_MAX_TURN_MS",
        "AA_TRANSPORT_TIMEOUT_S",
        "AA_QUALIFICATION_SCENARIO_WAIT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    policy = current_time_policy()
    assert policy.turn_time_limits_enabled is False
    assert policy.latency_gate_mode == "observe"
    assert policy.latency_p95_target_ms == 60_000.0
    assert policy.latency_max_turn_ms == 120_000.0
    assert policy.transport_timeout_s == 300.0
    assert policy.qualification_scenario_wait_s == 300.0
    settings = Settings.from_env({})
    assert settings.time_policy == policy


def test_invalid_time_policy_is_rejected() -> None:
    with pytest.raises(TimePolicyError):
        TimePolicy.from_env({"AA_TURN_TIME_LIMITS_ENABLED": "maybe"})
    with pytest.raises(TimePolicyError):
        TimePolicy.from_env({"AA_LATENCY_GATE_MODE": "diagnostic"})
    with pytest.raises(TimePolicyError):
        TimePolicy.from_env({"AA_LATENCY_P95_TARGET_MS": "0"})
    with pytest.raises(TimePolicyError):
        TimePolicy.from_env({"AA_LATENCY_MAX_TURN_MS": "10"})
    with pytest.raises(TimePolicyError):
        Settings.from_env({"AA_TRANSPORT_TIMEOUT_S": "-1"})


def test_disabled_budget_crosses_all_old_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "false")
    budget = new_turn_budget(upstream_latency_ms=5_000_000.0)
    assert budget.time_limits_enabled is False
    assert budget.remaining_s() == float("inf")
    assert budget.is_expired() is False
    assert budget.allocate_semantic_timeout_s() is None
    assert _retrieval_preserves_downstream(budget) is True
    assert _retrieval_semantic_stage_remaining_s(budget) == float("inf")

    async def _slow_but_successful() -> str:
        await asyncio.sleep(0)
        return "ok"

    result = asyncio.run(await_model_under_turn_budget(_slow_but_successful, budget))
    assert result == "ok"


def test_enabled_budget_keeps_old_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "true")
    budget = new_turn_budget(upstream_latency_ms=5_000_000.0)
    assert budget.time_limits_enabled is True
    with pytest.raises(TurnBudgetExpired):
        budget.allocate_semantic_timeout_s()
    assert _retrieval_preserves_downstream(budget) is False
    assert _retrieval_semantic_stage_remaining_s(budget) == 0.0


def test_observe_gate_e_publishes_slow_timing_without_slo_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AA_LATENCY_GATE_MODE", "observe")
    report = evaluate_latency_report([_sample(150_000.0), _sample(61_000.0)])
    assert report.status == "PASS"
    assert report.max_ms == 150_000.0
    assert report.detail == OBSERVE_DETAIL
    assert report.slo_evaluated is False
    ok, detail, metrics = slo_guards([150_000.0], mode="observe")
    assert ok is True
    assert detail == OBSERVE_DETAIL
    assert metrics["policy_mode"] == "observe"
    assert metrics["slo_evaluated"] is False


def test_enforce_gate_e_applies_thresholds_to_same_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AA_LATENCY_GATE_MODE", "enforce")
    assert evaluate_latency_report([_sample(60_000.0)]).status == "PASS"
    assert evaluate_latency_report([_sample(60_001.0)]).status == "FAIL"
    assert evaluate_latency_report([_sample(120_000.0)]).status == "FAIL"
    ok, detail, metrics = slo_guards([150_000.0], mode="enforce")
    assert ok is False
    assert "150000ms" in detail or "120000ms" in detail
    assert metrics["slo_evaluated"] is True


def test_censored_timeout_fails_reliability_in_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mode in ("observe", "enforce"):
        monkeypatch.setenv("AA_LATENCY_GATE_MODE", mode)
        report = evaluate_latency_report([_sample(900_000.0, outcome="timeout")])
        assert report.status == "FAIL"
        assert report.censored_count == 1
        assert report.max_ms == 900_000.0
    monkeypatch.setenv("AA_LATENCY_GATE_MODE", "observe")
    assert evaluate_latency_report([]).status == "BLOCKED"


def test_typed_live_samples_are_not_recapped_to_old_budget() -> None:
    payload = {
        "lanes": [
            {
                "lane": "live-telegram-evidence",
                "metrics": {
                    "turn_latency_samples": [
                        {
                            "causality_id": new_causality_id(),
                            "family": "ordinary-text",
                            "mode": "text",
                            "outcome": "served",
                            "latency_ms": 150_000.0,
                            "monotonic": True,
                            "clock": "monotonic-perf_counter",
                            "series": C_SERIES,
                            "boundary": "production-update-intake-to-simulated-handoff",
                            "receipt_kind": "simulated-in-process",
                            "censored": False,
                        }
                    ]
                },
            }
        ]
    }
    samples = build_c_series_from_live_summary(payload)
    assert samples[0].latency_ms == 150_000.0


def test_legacy_timeout_uses_observed_elapsed_without_policy_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AA_QUALIFICATION_SCENARIO_WAIT_S", "300")
    for observed_ms in (130_000.0, 500_000.0):
        payload = {
            "lanes": [
                {
                    "lane": "live-telegram-evidence",
                    "metrics": {
                        "scenario_diagnostics": [
                            {
                                "family": "ordinary-text",
                                "delivery_timeout": True,
                                "delivery_latency_ms": observed_ms,
                            }
                        ]
                    },
                }
            ]
        }
        samples = build_c_series_from_live_summary(payload)
        assert [sample.latency_ms for sample in samples] == [observed_ms]


def test_legacy_timeout_uses_recorded_wait_not_current_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AA_QUALIFICATION_SCENARIO_WAIT_S", "300")
    payload = {
        "lanes": [
            {
                "lane": "live-telegram-evidence",
                "failed": ["live-delivery-ordinary-text-timeout"],
                "metrics": {"qualification_scenario_wait_s": 180.0},
            }
        ]
    }
    samples = build_c_series_from_live_summary(payload)
    assert [sample.latency_ms for sample in samples] == [180_000.0]


def test_legacy_timeout_without_recorded_wait_preserves_bounded_censor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AA_QUALIFICATION_SCENARIO_WAIT_S", "300")
    payload = {
        "lanes": [
            {
                "lane": "live-telegram-evidence",
                "failed": ["live-delivery-ordinary-text-timeout"],
                "metrics": {},
            }
        ]
    }
    samples = build_c_series_from_live_summary(payload)
    assert [sample.latency_ms for sample in samples] == [120_000.0]


def test_script_gate_e_is_mode_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AA_LATENCY_GATE_MODE", "observe")
    observe = runner._gate_e(SHA, "run-1", PRODUCT, RUNTIME, [150_000.0], gate_c_live=True)
    assert observe.status == "PASS"
    assert observe.detail == OBSERVE_DETAIL
    monkeypatch.setenv("AA_LATENCY_GATE_MODE", "enforce")
    enforce = runner._gate_e(SHA, "run-1", PRODUCT, RUNTIME, [150_000.0], gate_c_live=True)
    assert enforce.status == "FAIL"


def test_gate_f_summary_and_label_make_scope_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AA_LATENCY_GATE_MODE", "observe")
    gates = tuple(
        GateEvidence(
            gate=gate,
            status="PASS",
            sha=SHA,
            product_fingerprint=PRODUCT,
            runtime_fingerprint=RUNTIME,
            run_id="run-1",
            live_trusted=gate in ("C", "D", "E"),
        )
        for gate in ("A", "B", "C", "E")
    )
    verdict = FinalVerdict(
        sha=SHA,
        status="PASS",
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        gates=gates,
        blocking_gate="",
        run_id="run-1",
    )
    summary = verdict_summary_v2(verdict, failures=[])
    assert summary["latency_gate_mode"] == "observe"
    assert summary["qualification_scope"] == "quality-first; latency SLO not evaluated"
    assert "quality-first; latency SLO not evaluated" in refusal_explanation(
        verdict, current_sha=SHA
    )


async def test_planner_slow_success_not_cut_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation import planner_node

    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "false")

    class _Model:
        wire_agent = "custom-agent"

        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, Any],
            retry_count: int = 2,
        ) -> dict[str, Any]:
            _ = (prompt, system, schema, retry_count)
            await asyncio.sleep(0.05)
            return {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": ["query one", "query two"],
            }

    plan = await planner_node.run_planner(
        "evening craving", model=_Model(), summary="", recent=[], time_budget_s=0.01
    )
    assert 1 <= len(plan.queries) <= 16


def test_semantic_work_plan_time_gates_follow_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.semantic_work_plan import (
        SemanticTurnDAG,
        new_work_plan,
        remaining_turn_s,
    )

    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "false")
    disabled = new_work_plan(SemanticTurnDAG())
    assert disabled.time_limits_enabled is False
    assert disabled.allows_provider_call(remaining_s=0.0, need_s=1.0) is True
    assert disabled.verifier_slice(0.0) is None
    assert disabled.judge_slice(0.0) == float("inf")
    assert remaining_turn_s(turn_started_monotonic=0.0) == float("inf")
    assert disabled.repair_full_round_allowed(full_rounds_used=2) is False

    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "true")
    enabled = new_work_plan(SemanticTurnDAG())
    assert enabled.time_limits_enabled is True
    assert enabled.allows_provider_call(remaining_s=0.0, need_s=1.0) is False
    assert enabled.verifier_slice(0.0) == 0.0
    assert enabled.judge_slice(10.0) == 7.0
    assert enabled.repair_full_round_allowed(full_rounds_used=2) is False
