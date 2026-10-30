"""Gate C focus mode result builders (kodmial/aa#431).

The transient Gate C focus mode runs exactly the 18 unchanged real-model
Gate C scenarios through the standard product-contract live runner, with
no Gate A/B execution and no trusted full reducer. This module builds the
typed focus artifacts:

- ``c-focus-result.json`` (schema ``aa-gate-c-focus-result/1``): the
  `PASS|FAIL|BLOCKED` status, exact SHA, run id, model identity, the
  failing stage and privacy-safe diagnostics. It is never a ``result.json``
  of the full Gate F, never a trusted ``continuum-qualification-result
  result=pass`` and never a ``QUALIFIED`` verdict; automaticalic closure of
  the qualification tracker stays out of reach for a C-only PASS.
- ``gate-E-diagnostic.json`` (schema ``aa-gate-e-diagnostic/1``): Gate E
  computed over the C samples as a DIAGNOSTIC metric only. It applies no
  SLO thresholds and carries no PASS/FAIL verdict; the full Gate E SLO
  tests/thresholds and its trusted artifact remain the province of full
  mode.

All payloads contain privacy-safe fields only: ids, digests, bounded
latencies, family/stage labels from the closed vocabulary; never prompts,
replies, book passages, session ids, or secrets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_C_FOCUS_RESULT = "aa-gate-c-focus-result/1"
SCHEMA_GATE_E_DIAGNOSTIC = "aa-gate-e-diagnostic/1"
SCHEMA_VERSION_FULL_RESULT = "aa-self-proving-verdict/2"

C_FOCUS_STATUS_PASS = "PASS"
C_FOCUS_STATUS_FAIL = "FAIL"
C_FOCUS_STATUS_BLOCKED = "BLOCKED"
C_FOCUS_STATUSES: tuple[str, ...] = (
    C_FOCUS_STATUS_PASS,
    C_FOCUS_STATUS_FAIL,
    C_FOCUS_STATUS_BLOCKED,
)

C_FOCUS_VERDICT_PASS = "C_PASS"
C_FOCUS_VERDICT_FAIL = "C_FAIL"
C_FOCUS_VERDICT_BLOCKED = "C_BLOCKED"
C_FOCUS_VERDICTS: tuple[str, ...] = (
    C_FOCUS_VERDICT_PASS,
    C_FOCUS_VERDICT_FAIL,
    C_FOCUS_VERDICT_BLOCKED,
)

_VERDICT_BY_STATUS = {
    C_FOCUS_STATUS_PASS: C_FOCUS_VERDICT_PASS,
    C_FOCUS_STATUS_FAIL: C_FOCUS_VERDICT_FAIL,
    C_FOCUS_STATUS_BLOCKED: C_FOCUS_VERDICT_BLOCKED,
}

_STATUS_BY_VERDICT = {
    C_FOCUS_VERDICT_PASS: C_FOCUS_STATUS_PASS,
    C_FOCUS_VERDICT_FAIL: C_FOCUS_STATUS_FAIL,
    C_FOCUS_VERDICT_BLOCKED: C_FOCUS_STATUS_BLOCKED,
}

_TRUSTED_FULL_GATE_STATUSES = frozenset({"PASS"})
_FAILURE_VERDICT_STATUSES = frozenset({"FAIL", "INCOMPLETE", "STALE"})


def _truncate(value: object, limit: int) -> str:
    return str(value or "")[:limit]


def status_for_verdict(verdict: str) -> str:
    """Map the root-boundary verdict to the typed result status."""
    if verdict not in (
        C_FOCUS_VERDICT_PASS,
        C_FOCUS_VERDICT_FAIL,
        C_FOCUS_VERDICT_BLOCKED,
    ):
        raise ValueError(f"unknown C-focus verdict: {verdict!r}")
    return _STATUS_BY_VERDICT[verdict]


def verdict_for_status(status: str) -> str:
    """Map one real Gate C verdict to the focus vocabulary (fail closed)."""
    normalized = str(status or "").strip().upper()
    if normalized in _TRUSTED_FULL_GATE_STATUSES:
        return C_FOCUS_VERDICT_PASS
    if normalized in {"FAIL", "INCOMPLETE"}:
        return C_FOCUS_VERDICT_FAIL
    # STALE/BLOCKED/unknown: infrastructure/identity cases stay BLOCKED,
    # never a product FAIL and never a PASS.
    return C_FOCUS_VERDICT_BLOCKED


def _live_lane_metrics(live_summary: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Locate the live-telegram-evidence lane metrics of a live summary."""
    if not isinstance(live_summary, Mapping):
        return {}
    lanes = live_summary.get("lanes")
    if not isinstance(lanes, Sequence):
        return {}
    for lane in lanes:
        if not isinstance(lane, Mapping):
            continue
        if str(lane.get("lane", "")) == "live-telegram-evidence":
            metrics = lane.get("metrics")
            if isinstance(metrics, Mapping):
                return metrics
    return {}


def _failing_stage(live_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    """Privacy-safe stage of the last failed scenario (diagnostics only)."""
    metrics = _live_lane_metrics(live_summary)
    diagnostics = metrics.get("scenario_diagnostics")
    if not isinstance(diagnostics, Sequence):
        return {"stage_of_failure": ""}
    for entry in reversed(list(diagnostics)):
        if not isinstance(entry, Mapping):
            continue
        if bool(entry.get("delivery_timeout")) or str(entry.get("timeout_reason", "")):
            return {
                "stage_of_failure": _truncate(entry.get("last_active_stage"), 64),
                "timeout_category": _truncate(entry.get("timeout_category"), 32),
                "timeout_reason": _truncate(entry.get("timeout_reason"), 32),
                "last_request_state": _truncate(entry.get("request_state"), 48),
                "stall_kind": _truncate(entry.get("stall_kind"), 48),
                "failed_scenario_index": entry.get("scenario_index"),
                "failed_scenario_family": _truncate(entry.get("family", ""), 64),
            }
    return {"stage_of_failure": ""}


def _latency_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    try:
        samples = metrics.get("turn_latency_samples")
        values: list[float] = []
        if isinstance(samples, Sequence):
            for sample in samples:
                if isinstance(sample, Mapping):
                    latency = sample.get("latency_ms")
                    if isinstance(latency, (int, float)) and latency > 0:
                        values.append(float(latency))
        if not values:
            return {"latency_sample_count": 0}
        return {
            "latency_sample_count": len(values),
            "latency_p50_ms": round(percentile(values, 50), 1),
            "latency_p95_ms": round(percentile(values, 95), 1),
            "latency_max_ms": round(max(values), 1),
        }
    except Exception:
        return {"latency_sample_count": 0}


def percentile(values: Sequence[float], pct: float) -> float:
    """Bounded percentile over the provided latency values (ms)."""
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil((pct / 100.0) * len(ordered))) - 1))
    return ordered[index]


def build_c_focus_result(
    *,
    main_sha: str,
    run_id: str,
    gate_c_evidence: Mapping[str, Any] | None,
    live_summary: Mapping[str, Any] | None = None,
    model_identity: str = "",
    bootstrap_status: str = "",
) -> dict[str, Any]:
    """Build the ``aa-gate-c-focus-result/1`` payload from real evidence.

    Missing/inconsistent Gate C evidence is BLOCKED (fail closed); a real
    PASS from Gate C maps to the C_PASS verdict and remains explicitly
    NON-product: ``product_qualified=false`` always holds in focus mode.
    """
    if not isinstance(gate_c_evidence, Mapping):
        return _blocked_result(
            main_sha=main_sha,
            run_id=run_id,
            model_identity=model_identity,
            bootstrap_status=bootstrap_status,
            reasoning="gate-c-evidence-missing",
        )
    sha_mismatch = str(gate_c_evidence.get("sha", "")).lower() != str(main_sha).lower()
    if sha_mismatch:
        return _blocked_result(
            main_sha=main_sha,
            run_id=run_id,
            model_identity=model_identity,
            bootstrap_status=bootstrap_status,
            reasoning="gate-c-evidence-sha-mismatch",
        )
    gate_status = str(gate_c_evidence.get("status", "")).strip().upper()
    verdict = verdict_for_status(gate_status)
    status = status_for_verdict(verdict)
    identity = _truncate(model_identity, 128)
    if not identity:
        metrics = _live_lane_metrics(live_summary)
        identity = _truncate(metrics.get("latency_provider_id"), 128)
    diagnostics: dict[str, Any] = {
        "gate_c_status": gate_status or "missing",
        "gate_c_failure_category": _truncate(gate_c_evidence.get("failure_category"), 64),
        "gate_c_component": _truncate(gate_c_evidence.get("component"), 64),
        "scenario_mode": "18-real-model-scenarios",
        "a_b_executed": False,
        "product_qualified": False,
    }
    metrics = _live_lane_metrics(live_summary)
    if metrics:
        diagnostics.update(_latency_summary(metrics))
        # kodmial/aa#453: a watchdog-killed lane carries the explicit
        # per-source coverage of its partial diagnostics; the c-focus
        # artifact must state what was observed vs missing/unobserved
        # instead of implicating silence as absent product behavior.
        diagnostics["partial_evidence_reported"] = bool(
            metrics.get("partial_evidence_reported", False)
        )
        diagnostics["watchdog_diagnostics_status"] = _truncate(
            metrics.get("watchdog_diagnostics_status", ""), 64
        )
        coverage = metrics.get("partial_evidence_coverage")
        if isinstance(coverage, Mapping):
            diagnostics["partial_evidence_coverage"] = {
                _truncate(name, 64): _truncate(state, 32) for name, state in coverage.items()
            }
        diagnostics["independent_judge_unavailable"] = int(
            metrics.get("independent_judge_unavailable", 0) or 0
        )
        diagnostics["book_grounded_expected"] = metrics.get("book_grounded_expected")
        diagnostics["book_grounded_successes"] = metrics.get("book_grounded_successes")
    diagnostics.update(_failing_stage(live_summary))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_C_FOCUS_RESULT,
        "mode": "gate-c-focus",
        "verdict": verdict,
        "status": status,
        "sha": _truncate(main_sha, 64),
        "run_id": _truncate(run_id, 64),
        "model_identity": identity,
        "not_qualified": "c-only PASS is not a full trusted qualification",
    }
    if bootstrap_status:
        result["bootstrap_status"] = _truncate(bootstrap_status, 64)
    result["privacy_safe_diagnostics"] = diagnostics
    return result


def _blocked_result(
    *,
    main_sha: str,
    run_id: str,
    model_identity: str,
    bootstrap_status: str,
    reasoning: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_C_FOCUS_RESULT,
        "mode": "gate-c-focus",
        "verdict": C_FOCUS_VERDICT_BLOCKED,
        "status": C_FOCUS_STATUS_BLOCKED,
        "sha": _truncate(main_sha, 64),
        "run_id": _truncate(run_id, 64),
        "model_identity": _truncate(model_identity, 128),
        "not_qualified": "c-only PASS is not a full trusted qualification",
        "privacy_safe_diagnostics": {
            "block_reason": _truncate(reasoning, 64),
            "a_b_executed": False,
            "product_qualified": False,
        },
    }
    if bootstrap_status:
        result["bootstrap_status"] = _truncate(bootstrap_status, 64)
    return result


def build_gate_e_diagnostic(
    *,
    main_sha: str,
    run_id: str,
    live_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the Gate E diagnostic metric from the C latency samples.

    p50/p95/max plus reliability counts; no threshold evaluation and no
    verdict — the full Gate E SLO tests remain untouched in full mode.
    """
    metrics = _live_lane_metrics(live_summary)
    samples = metrics.get("turn_latency_samples")
    values: list[float] = []
    censored_count = 0
    served_count = 0
    product_failure_count = 0
    delivered_service_error_count = 0
    try:
        from aa.qualification.gate_latency import PRODUCT_FAILURE_OUTCOMES

        product_failure_vocab = PRODUCT_FAILURE_OUTCOMES
    except Exception:  # pragma: no cover - defensive import fallback
        product_failure_vocab = frozenset({"service_error", "rejected"})
    if isinstance(samples, Sequence):
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            latency = sample.get("latency_ms")
            if isinstance(latency, (int, float)) and float(latency) > 0:
                values.append(float(latency))
            if bool(sample.get("censored")):
                censored_count += 1
            if str(sample.get("outcome", "")) == "served":
                served_count += 1
            # kodmial/aa#436: delivered service errors and other product
            # failures are counted separately from latency-only outcome;
            # legacy samples (no product fields) stay inconclusive.
            product_outcome = str(sample.get("product_outcome", ""))
            if product_outcome in product_failure_vocab:
                product_failure_count += 1
            if product_outcome == "service_error":
                delivered_service_error_count += 1
    diagnostic: dict[str, Any] = {
        "schema_version": SCHEMA_GATE_E_DIAGNOSTIC,
        "mode": "diagnostic-metric-on-c-samples",
        "sha": _truncate(main_sha, 64),
        "run_id": _truncate(run_id, 64),
        "verdicts_applicable": "none (diagnostic metric; full Gate E SLOs untouched)",
        "slo_thresholds_applied": False,
        "latency_sample_count": len(values),
        "served_turn_count": served_count,
        "censored_turn_count": censored_count,
        "product_failure_turn_count": product_failure_count,
        "delivered_service_error_count": delivered_service_error_count,
        "latency_boundary": _truncate(
            metrics.get("latency_boundary", "production-update-intake-to-simulated-handoff"), 64
        ),
        "latency_series": _truncate(metrics.get("latency_series"), 64),
    }
    if values:
        diagnostic["latency_p50_ms"] = round(percentile(values, 50), 1)
        diagnostic["latency_p95_ms"] = round(percentile(values, 95), 1)
        diagnostic["latency_max_ms"] = round(max(values), 1)
        diagnostic["latency_units"] = "ms"
    else:
        diagnostic["latency_p50_ms"] = None
        diagnostic["latency_p95_ms"] = None
        diagnostic["latency_max_ms"] = None
        diagnostic["non_applicable_reason"] = "no-c-latency-samples"
    return diagnostic


__all__ = [
    "C_FOCUS_STATUSES",
    "C_FOCUS_VERDICTS",
    "SCHEMA_C_FOCUS_RESULT",
    "SCHEMA_GATE_E_DIAGNOSTIC",
    "build_c_focus_result",
    "build_gate_e_diagnostic",
    "percentile",
    "status_for_verdict",
    "verdict_for_status",
]
