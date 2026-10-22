"""Gate E latency ownership and Gate F trust inputs (kodmial/aa#342).

Gate E is an independent evidence consumer and latency calculator. It never
repeats Gate C, the A/B index bootstrap, provider calls, or Gate D live
network operations. It consumes already-produced, validated evidence:

- ``C-series`` (mandatory): raw production-update intake to confirmed
  simulated-network handoff measured on the true production graph with the
  real OpenCode/provider/book path. Monotonic clock, one causality id per
  turn, simulated receipt only.
- ``D-runtime`` (mandatory lane): real Telegram identity, polling/startup
  and readiness timings from Gate D evidence. Never synthesized from C.
- ``external-optional``: actual network send/voice acknowledgment and full
  live delivered-turn latency, only when a safely isolated bot plus an
  explicit consented test peer exist. Missing optional peer is recorded
  ``external_send_unverified`` and is never a mandatory D failure.

Failed/timeout ordinary turns are counted in reliability and bounded-censored
latency metrics, never dropped to improve the SLO. Simulated ack is never
relabeled as actual network delivery, and full real end-to-end SLO is never
inferred by adding unrelated p95s.

User-facing SLO rules are unchanged: p95 <= 60s, no ordinary text turn >=
120s (voice budgets are distinct and read from the contract, never reused
implicitly from the text threshold).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

from aa.qualification.self_proving import (
    ORDINARY_TURN_BUDGET_MS,
    GateEvidence,
    percentile_ms,
)
from aa.time_policy import (
    TimePolicy,
    current_time_policy,
    evaluate_latency_thresholds,
)

LATENCY_EVIDENCE_SCHEMA = "aa-gate-e-latency/1"
CHECKPOINT_SCHEMA = "aa-gate-checkpoint/1"

C_SERIES = "c-series-simulated-handoff"
D_RUNTIME_SERIES = "d-runtime-readiness"
EXTERNAL_OPTIONAL_SERIES = "external-optional-delivery"

CLOCK_MONOTONIC = "monotonic-perf_counter"
UNITS_MS = "ms"

MISSING_LIVE_SAMPLES = "missing-live-samples"
# kodmial/aa#417 Gate E honesty: when real bounded-censored samples exist
# but the C lane cannot hand them to the SLO as trustworthy success
# evidence, the blocked category must not claim that samples are missing.
INSUFFICIENT_COMPLETED_TURNS = "insufficient-completed-turns"
SLO_EXCEEDED = "latency-budget-exceeded"
RELIABILITY_FAILED = "ordinary-turn-failed"

EXTERNAL_CONFIRMED = "external_send_confirmed"
EXTERNAL_UNVERIFIED = "external_send_unverified"

VALID_SAMPLE_OUTCOMES = ("served", "timeout", "failed")
VALID_SAMPLE_MODES = ("text", "voice")
VALID_SERIES = (C_SERIES, D_RUNTIME_SERIES, EXTERNAL_OPTIONAL_SERIES)


class GateLatencyError(ValueError):
    """Raised when latency evidence handling fails closed."""


def new_causality_id() -> str:
    """Return a fresh opaque causality id (no user/corpus text)."""
    return uuid.uuid4().hex


def _as_finite_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _is_finite_positive(value: Any) -> bool:
    return _as_finite_positive_float(value) is not None


@dataclass(frozen=True)
class TurnSample:
    """One measured ordinary turn (privacy-safe, no text)."""

    causality_id: str
    family: str
    mode: str
    outcome: str
    latency_ms: float
    monotonic: bool = True
    clock: str = CLOCK_MONOTONIC
    series: str = C_SERIES
    boundary: str = "production-update-intake-to-simulated-handoff"
    receipt_kind: str = "simulated-in-process"
    censored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "causality_id": self.causality_id,
            "family": self.family,
            "mode": self.mode,
            "outcome": self.outcome,
            "latency_ms": float(self.latency_ms),
            "monotonic": bool(self.monotonic),
            "clock": self.clock,
            "series": self.series,
            "boundary": self.boundary,
            "receipt_kind": self.receipt_kind,
            "censored": bool(self.censored),
        }

    @staticmethod
    def from_dict(payload: Any) -> TurnSample:
        if not isinstance(payload, dict):
            raise GateLatencyError("turn sample must be an object")
        causality_id = str(payload.get("causality_id", "") or "").strip()
        if not causality_id or len(causality_id) > 64:
            raise GateLatencyError("turn sample needs a causality id")
        family = str(payload.get("family", "") or "").strip()[:64] or "unknown"
        mode = str(payload.get("mode", "text") or "text").strip()
        if mode not in VALID_SAMPLE_MODES:
            raise GateLatencyError(f"unknown sample mode {mode!r}")
        outcome = str(payload.get("outcome", "served") or "served").strip()
        if outcome not in VALID_SAMPLE_OUTCOMES:
            raise GateLatencyError(f"unknown sample outcome {outcome!r}")
        try:
            latency_ms = float(payload.get("latency_ms", 0.0))
        except (TypeError, ValueError) as exc:
            raise GateLatencyError("sample latency must be numeric") from exc
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise GateLatencyError("sample latency must be finite positive")
        series = str(payload.get("series", C_SERIES) or C_SERIES).strip()
        if series not in VALID_SERIES:
            raise GateLatencyError(f"unknown latency series {series!r}")
        receipt_kind = str(payload.get("receipt_kind", "simulated-in-process") or "")
        if series == C_SERIES and receipt_kind not in ("simulated-in-process", "none"):
            raise GateLatencyError("C-series receipt must stay simulated")
        if series == EXTERNAL_OPTIONAL_SERIES and receipt_kind not in (
            "external-confirmed",
            "external-unverified",
        ):
            raise GateLatencyError("optional series needs an external receipt label")
        clock = str(payload.get("clock", CLOCK_MONOTONIC) or CLOCK_MONOTONIC).strip()
        monotonic = bool(payload.get("monotonic", True))
        if clock != CLOCK_MONOTONIC or not monotonic:
            raise GateLatencyError("latency samples require a monotonic clock")
        boundary = str(payload.get("boundary", "") or "").strip()[:128]
        censored = bool(payload.get("censored", outcome in ("timeout", "failed")))
        if outcome in ("timeout", "failed") and not censored:
            censored = True
        return TurnSample(
            causality_id=causality_id,
            family=family,
            mode=mode,
            outcome=outcome,
            latency_ms=float(latency_ms),
            monotonic=True,
            clock=CLOCK_MONOTONIC,
            series=series,
            boundary=boundary,
            receipt_kind=receipt_kind,
            censored=censored,
        )


@dataclass(frozen=True)
class LatencyReport:
    """Evaluated Gate E latency report (aggregate, privacy-safe)."""

    status: str
    sample_count: int
    served_count: int
    censored_count: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    units: str = UNITS_MS
    boundary: str = "production-update-intake-to-simulated-handoff"
    provider_id: str = ""
    source_proof_kind: str = "c-series-simulated-handoff"
    failure_category: str = ""
    detail: str = ""
    latency_gate_mode: str = "observe"
    slo_evaluated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LATENCY_EVIDENCE_SCHEMA,
            "status": self.status,
            "sample_count": int(self.sample_count),
            "served_count": int(self.served_count),
            "censored_count": int(self.censored_count),
            "p50_ms": float(self.p50_ms),
            "p95_ms": float(self.p95_ms),
            "max_ms": float(self.max_ms),
            "units": self.units,
            "boundary": self.boundary,
            "provider_id": self.provider_id,
            "source_proof_kind": self.source_proof_kind,
            "failure_category": self.failure_category,
            "detail": self.detail,
            "latency_gate_mode": self.latency_gate_mode,
            "slo_evaluated": bool(self.slo_evaluated),
        }


def validate_samples(samples: list[TurnSample]) -> None:
    """Fail closed on duplicate causality ids or non-monotonic samples."""
    seen: set[str] = set()
    for sample in samples:
        if sample.causality_id in seen:
            raise GateLatencyError("duplicate causality id")
        seen.add(sample.causality_id)
        if not sample.monotonic or sample.clock != CLOCK_MONOTONIC:
            raise GateLatencyError("non-monotonic latency sample")
        if not math.isfinite(sample.latency_ms) or sample.latency_ms <= 0:
            raise GateLatencyError("invalid latency sample")


def coerce_samples(raw: Any, *, series: str = C_SERIES) -> list[TurnSample]:
    """Coerce a raw sample list into validated TurnSamples (fail-closed)."""
    if not isinstance(raw, list) or not raw:
        raise GateLatencyError(MISSING_LIVE_SAMPLES)
    samples = [TurnSample.from_dict(entry) for entry in raw]
    for sample in samples:
        if sample.series != series and series == C_SERIES and sample.series != C_SERIES:
            raise GateLatencyError("mixed latency series are non-interchangeable")
    validate_samples(samples)
    return samples


def samples_from_legacy_latencies(
    latencies_ms: list[float],
    *,
    family: str = "ordinary-text",
    mode: str = "text",
) -> list[TurnSample]:
    """Wrap legacy per-turn float latencies as C-series served samples."""
    samples: list[TurnSample] = []
    for value in latencies_ms:
        if not _is_finite_positive(value):
            raise GateLatencyError("invalid latency sample")
        samples.append(
            TurnSample(
                causality_id=new_causality_id(),
                family=family,
                mode=mode,
                outcome="served",
                latency_ms=float(value),
                series=C_SERIES,
            )
        )
    if not samples:
        raise GateLatencyError(MISSING_LIVE_SAMPLES)
    return samples


def _diagnostic_timeout_latency_ms(entry: dict[str, Any]) -> float | None:
    """Return an actually observed timeout latency from a legacy diagnostic.

    Legacy payloads may carry the elapsed wait/deadline in one of these
    privacy-safe numeric fields. Prefer it over the configured operational
    bound so censored evidence never claims more elapsed time than was
    actually observed.
    """
    for key in (
        "delivery_latency_ms",
        "observed_latency_ms",
        "elapsed_ms",
        "latency_ms",
    ):
        numeric = _as_finite_positive_float(entry.get(key))
        if numeric is not None:
            return numeric
    return None


def build_c_series_from_live_summary(
    payload: dict[str, Any],
    *,
    run_id: str = "",
) -> list[TurnSample]:
    """Build the mandatory C-series from a live-out summary (no synthesis).

    Served turns come from ``turn_latencies_ms`` plus per-scenario
    ``delivery_latency_ms`` diagnostics. Delivery timeouts recorded in lane
    ``failed`` entries and ``scenario_diagnostics`` with
    ``delivery_timeout=true`` become bounded-censored timeout samples at the
    observed deadline instead of being dropped.
    """
    _ = run_id
    if not isinstance(payload, dict):
        raise GateLatencyError(MISSING_LIVE_SAMPLES)
    lanes = payload.get("lanes", [])
    if not isinstance(lanes, list):
        raise GateLatencyError(MISSING_LIVE_SAMPLES)
    live_lane: dict[str, Any] | None = None
    for lane in lanes:
        if isinstance(lane, dict) and str(lane.get("lane", "")) == "live-telegram-evidence":
            live_lane = lane
            break
    if live_lane is None:
        raise GateLatencyError(MISSING_LIVE_SAMPLES)
    metrics = live_lane.get("metrics", {})
    if not isinstance(metrics, dict):
        raise GateLatencyError(MISSING_LIVE_SAMPLES)
    # New payloads record the scenario wait actually used for the run.
    # Legacy payloads predate that field, so fall back to the old
    # bounded-censor value instead of the current process policy: rebuilding
    # old evidence under a larger configured wait must not inflate it.
    recorded_wait_s = _as_finite_positive_float(metrics.get("qualification_scenario_wait_s"))
    recorded_wait_ms = recorded_wait_s * 1000.0 if recorded_wait_s is not None else None
    legacy_bound_ms = float(ORDINARY_TURN_BUDGET_MS)
    typed_raw = metrics.get("turn_latency_samples", [])
    if isinstance(typed_raw, list) and typed_raw:
        typed: list[TurnSample] = []
        for entry in typed_raw:
            if not isinstance(entry, dict):
                raise GateLatencyError("turn sample must be an object")
            outcome = str(entry.get("outcome", "served") or "served").strip()
            if outcome == "harness-error":
                typed.append(
                    TurnSample(
                        causality_id=new_causality_id(),
                        family=str(entry.get("family", "") or "ordinary-text")[:64],
                        mode="text",
                        outcome="failed",
                        latency_ms=float(entry.get("latency_ms", 0.0) or 0.0),
                        series=C_SERIES,
                        censored=True,
                    )
                )
                continue
            typed.append(TurnSample.from_dict(entry))
        validate_samples(typed)
        return typed
    served: list[TurnSample] = []
    raw_served = metrics.get("turn_latencies_ms", [])
    if isinstance(raw_served, list):
        for value in raw_served:
            if not _is_finite_positive(value):
                continue
            served.append(
                TurnSample(
                    causality_id=new_causality_id(),
                    family="ordinary-text",
                    mode="text",
                    outcome="served",
                    latency_ms=float(value),
                    series=C_SERIES,
                )
            )
    censored: list[TurnSample] = []
    diagnostics = metrics.get("scenario_diagnostics", [])
    if isinstance(diagnostics, list):
        for entry in diagnostics:
            if not isinstance(entry, dict):
                continue
            if bool(entry.get("delivery_timeout", False)) is not True:
                continue
            family = str(entry.get("family", "") or "ordinary-text")[:64]
            # Prefer the actual observed timeout elapsed time when the
            # legacy diagnostic carries it, bounded only by the operational
            # wait recorded for that run (if any). Never cap observed
            # elapsed time at a product SLO or the current process policy.
            observed_ms = _diagnostic_timeout_latency_ms(entry)
            if observed_ms is not None:
                latency_ms = (
                    min(observed_ms, recorded_wait_ms)
                    if recorded_wait_ms is not None
                    else observed_ms
                )
            else:
                latency_ms = recorded_wait_ms if recorded_wait_ms is not None else legacy_bound_ms
            censored.append(
                TurnSample(
                    causality_id=new_causality_id(),
                    family=family or "ordinary-text",
                    mode="text",
                    outcome="timeout",
                    latency_ms=latency_ms,
                    series=C_SERIES,
                    censored=True,
                )
            )
    # Lane-level timeout failures without a diagnostics row still count.
    failed = live_lane.get("failed", [])
    if isinstance(failed, list):
        timeout_marks = sum(
            1 for name in failed if isinstance(name, str) and "timeout" in name.lower()
        )
        extra = max(0, timeout_marks - len(censored))
        for _ in range(extra):
            censored.append(
                TurnSample(
                    causality_id=new_causality_id(),
                    family="ordinary-text",
                    mode="text",
                    outcome="timeout",
                    latency_ms=(
                        recorded_wait_ms if recorded_wait_ms is not None else legacy_bound_ms
                    ),
                    series=C_SERIES,
                    censored=True,
                )
            )
    samples = served + censored
    if not samples:
        raise GateLatencyError(MISSING_LIVE_SAMPLES)
    validate_samples(samples)
    return samples


def evaluate_latency_report(
    samples: list[TurnSample],
    *,
    provider_id: str = "",
    source_proof_kind: str = "c-series-simulated-handoff",
    boundary: str = "production-update-intake-to-simulated-handoff",
    mode: str | None = None,
    policy: TimePolicy | None = None,
) -> LatencyReport:
    """Evaluate C-series samples under the configured Gate E policy.

    Reliability is always enforced: any censored timeout/failed ordinary
    turn fails Gate E while still contributing its observed latency to
    p50/p95/max. In ``observe`` mode the p95/max thresholds are not
    evaluated; in ``enforce`` mode they are. Evidence-integrity failures
    remain BLOCKED in both modes.
    """
    active = policy if policy is not None else current_time_policy()
    if mode is not None:
        active = active.with_latency_mode(mode)
    if not samples:
        return LatencyReport(
            status="BLOCKED",
            sample_count=0,
            served_count=0,
            censored_count=0,
            p50_ms=0.0,
            p95_ms=0.0,
            max_ms=0.0,
            boundary=boundary,
            provider_id=provider_id[:64],
            source_proof_kind=source_proof_kind[:64],
            failure_category=MISSING_LIVE_SAMPLES,
            detail="no trustworthy live turn samples",
            latency_gate_mode=active.latency_gate_mode,
            slo_evaluated=active.latency_gate_mode == "enforce",
        )
    validate_samples(samples)
    latencies = [float(sample.latency_ms) for sample in samples]
    served_count = sum(1 for sample in samples if sample.outcome == "served")
    censored_count = sum(1 for sample in samples if sample.censored)
    p50 = percentile_ms(latencies, 50)
    p95 = percentile_ms(latencies, 95)
    maximum = max(latencies)

    def _report(status: str, *, failure_category: str = "", detail: str = "") -> LatencyReport:
        return LatencyReport(
            status=status,
            sample_count=len(samples),
            served_count=served_count,
            censored_count=censored_count,
            p50_ms=p50,
            p95_ms=p95,
            max_ms=maximum,
            boundary=boundary,
            provider_id=provider_id[:64],
            source_proof_kind=source_proof_kind[:64],
            failure_category=failure_category,
            detail=detail,
            latency_gate_mode=active.latency_gate_mode,
            slo_evaluated=active.latency_gate_mode == "enforce",
        )

    if censored_count > 0:
        return _report(
            "FAIL",
            failure_category=RELIABILITY_FAILED,
            detail=f"{censored_count} ordinary turn(s) timed out/failed; observed timing counted",
        )
    ok, detail = evaluate_latency_thresholds(p95, maximum, policy=active)
    if not ok:
        return _report("FAIL", failure_category=SLO_EXCEEDED, detail=detail)
    return _report("PASS", detail=detail)


def classify_optional_external_send(
    *,
    peer_configured: bool,
    consent_opt_in: bool,
    message_id: int | None = None,
    delivery_confirmed: bool = False,
) -> str:
    """Classify the optional external send (never a mandatory verdict)."""
    if not (peer_configured and consent_opt_in):
        return EXTERNAL_UNVERIFIED
    if not delivery_confirmed:
        return EXTERNAL_UNVERIFIED
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
        return EXTERNAL_UNVERIFIED
    return EXTERNAL_CONFIRMED


def latency_evidence_checksum(report: LatencyReport, *, sha: str, run_id: str) -> str:
    """Checksum over the immutable latency report body."""
    canonical = json.dumps(
        {
            "schema_version": LATENCY_EVIDENCE_SCHEMA,
            "status": report.status,
            "sample_count": report.sample_count,
            "served_count": report.served_count,
            "censored_count": report.censored_count,
            "p50_ms": report.p50_ms,
            "p95_ms": report.p95_ms,
            "max_ms": report.max_ms,
            "units": report.units,
            "boundary": report.boundary,
            "provider_id": report.provider_id,
            "source_proof_kind": report.source_proof_kind,
            "latency_gate_mode": report.latency_gate_mode,
            "slo_evaluated": bool(report.slo_evaluated),
            "sha": sha,
            "run_id": run_id,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_gate_e_evidence(
    report: LatencyReport,
    *,
    sha: str,
    run_id: str,
    product_fingerprint: str,
    runtime_fingerprint: str,
    provider_id: str = "",
    external_send_status: str = EXTERNAL_UNVERIFIED,
) -> GateEvidence:
    """Convert a latency report into Gate E evidence (single owner)."""
    _ = provider_id
    _ = external_send_status
    if report.status == "BLOCKED":
        return GateEvidence(
            gate="E",
            status="BLOCKED",
            sha=sha,
            product_fingerprint=product_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            failure_category=MISSING_LIVE_SAMPLES,
            component="slo",
            run_id=run_id,
            detail=report.detail[:160],
        )
    if report.status == "FAIL":
        return GateEvidence(
            gate="E",
            status="FAIL",
            sha=sha,
            product_fingerprint=product_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            failure_category=report.failure_category or SLO_EXCEEDED,
            component="slo",
            run_id=run_id,
            live_trusted=True,
            latency_p50_ms=report.p50_ms,
            latency_p95_ms=report.p95_ms,
            max_turn_ms=report.max_ms,
            detail=report.detail[:160],
        )
    return GateEvidence(
        gate="E",
        status="PASS",
        sha=sha,
        product_fingerprint=product_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        component="slo",
        run_id=run_id,
        live_trusted=True,
        latency_p50_ms=report.p50_ms,
        latency_p95_ms=report.p95_ms,
        max_turn_ms=report.max_ms,
        detail=report.detail[:160],
    )


def monotonic_now_ms() -> float:
    """Return a monotonic timestamp in milliseconds (single seam)."""
    return float(time.perf_counter() * 1000.0)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CLOCK_MONOTONIC",
    "C_SERIES",
    "D_RUNTIME_SERIES",
    "EXTERNAL_CONFIRMED",
    "EXTERNAL_OPTIONAL_SERIES",
    "EXTERNAL_UNVERIFIED",
    "LATENCY_EVIDENCE_SCHEMA",
    "INSUFFICIENT_COMPLETED_TURNS",
    "MISSING_LIVE_SAMPLES",
    "RELIABILITY_FAILED",
    "SLO_EXCEEDED",
    "UNITS_MS",
    "GateLatencyError",
    "LatencyReport",
    "TurnSample",
    "build_c_series_from_live_summary",
    "build_gate_e_evidence",
    "classify_optional_external_send",
    "coerce_samples",
    "evaluate_latency_report",
    "latency_evidence_checksum",
    "monotonic_now_ms",
    "new_causality_id",
    "samples_from_legacy_latencies",
    "validate_samples",
]
