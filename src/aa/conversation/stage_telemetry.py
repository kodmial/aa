"""Privacy-safe per-stage turn telemetry (issue #146, Gates C/E).

Gate C must record per-stage outcomes and latency for the exact
production conversation path without logging user or corpus text::

    application -> dispatcher -> langgraph -> planner -> retrieval ->
    answer -> verifier -> delivery

Gate E consumes the same samples for p50/p95 SLO guards and typing
heartbeat continuity. Diversity detection consumes reply signatures:
unrelated valid inputs must not collapse to one generic fallback (the
runtime #37422302821 failure mode).

Privacy: this module stores only stage names, boolean outcomes, latency
milliseconds, reply lengths, reply signature hashes, failure categories
and model identity. It never stores user text, evidence text, transcripts,
summaries, prompts or secrets.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from aa.qualification.self_proving import (
    GATE_C_STAGES,
    ORDINARY_TURN_BUDGET_MS,
    SelfProvingError,
    assert_no_text_leak,
    diversity_passes,
    heartbeat_continuity_ok,
    percentile_ms,
    slo_guards,
)

STAGES: tuple[str, ...] = GATE_C_STAGES

# Legacy optional stage: recorded repairs may appear in stored telemetry but
# are never required for Gate C coverage (GATE_C_STAGES is authoritative).
OPTIONAL_STAGES: tuple[str, ...] = ("repair",)


def reply_signature(reply: str) -> str:
    """Return the privacy-safe signature for one reply (never the text).

    Normalizes whitespace/case and hashes, so equality means the same
    generic fallback while different natural replies differ. Callers must
    never persist ``reply`` itself alongside the signature.
    """
    normalized = " ".join((reply or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class StageSample:
    """One privacy-safe stage outcome (no user/corpus text)."""

    stage: str
    ok: bool
    latency_ms: float
    category: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "latency_ms": round(float(self.latency_ms), 3),
            "category": self.category,
            "model": self.model,
        }


@dataclass
class TurnTelemetry:
    """Privacy-safe telemetry for one qualification turn."""

    family: str
    stages: list[StageSample] = field(default_factory=list)
    reply_len: int = 0
    reply_signature: str = ""
    fallback: bool = False
    served_model: str = ""

    def total_ms(self) -> float:
        """Return the summed per-stage latency for this turn."""
        return float(sum(item.latency_ms for item in self.stages))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "family": self.family,
            "stages": [item.to_dict() for item in self.stages],
            "reply_len": int(self.reply_len),
            "reply_signature": self.reply_signature,
            "fallback": bool(self.fallback),
            "served_model": self.served_model,
            "total_ms": round(self.total_ms(), 3),
        }
        assert_no_text_leak(payload, owner="turn-telemetry")
        return payload


def record_stage(
    telemetry: TurnTelemetry,
    *,
    stage: str,
    ok: bool,
    latency_ms: float,
    category: str = "",
    model: str = "",
) -> None:
    """Append one stage sample (validates stage name and latency)."""
    normalized = (stage or "").strip().lower()
    if normalized not in STAGES and normalized not in OPTIONAL_STAGES:
        raise SelfProvingError(f"unknown telemetry stage {stage!r}")
    try:
        value = float(latency_ms)
    except (TypeError, ValueError) as exc:
        raise SelfProvingError("stage latency must be finite and >= 0") from exc
    if not math.isfinite(value) or value < 0:
        raise SelfProvingError("stage latency must be finite and >= 0")
    telemetry.stages.append(
        StageSample(
            stage=normalized,
            ok=bool(ok),
            latency_ms=value,
            category=(category or "").strip()[:64],
            model=(model or "").strip()[:128],
        )
    )


def stage_totals(telemetries: list[TurnTelemetry]) -> dict[str, float]:
    """Return summed per-stage latency across turns (ms, privacy-safe)."""
    totals: dict[str, float] = {stage: 0.0 for stage in (*STAGES, *OPTIONAL_STAGES)}
    for turn in telemetries:
        for sample in turn.stages:
            if sample.stage in totals:
                totals[sample.stage] += float(sample.latency_ms)
    return totals


def evaluate_gate_c_telemetry(
    telemetries: list[TurnTelemetry], *, required_families: int = 4
) -> tuple[bool, str, dict[str, Any]]:
    """Evaluate Gate C telemetry: stages covered + diversity, no text."""
    if len(telemetries) < required_families:
        return (
            False,
            "too few scenario families executed",
            {
                "turns": len(telemetries),
                "required_families": required_families,
            },
        )
    families = {turn.family for turn in telemetries}
    if len(families) < required_families:
        return (
            False,
            "scenario families lack coverage",
            {
                "families": sorted(families),
                "required_families": required_families,
            },
        )
    for turn in telemetries:
        covered = {sample.stage for sample in turn.stages}
        missing = [stage for stage in STAGES if stage not in covered]
        verifier_failed = any(s.stage == "verifier" and not s.ok for s in turn.stages)
        if missing:
            return (
                False,
                f"family {turn.family} misses stages {','.join(missing)}",
                {
                    "family": turn.family,
                    "missing": missing,
                },
            )
        failed = [sample.stage for sample in turn.stages if not sample.ok]
        repair_idx = max(
            [i for i, s in enumerate(turn.stages) if s.stage == "repair" and s.ok],
            default=-1,
        )
        verifier_repassed = False
        if repair_idx >= 0:
            verifier_repassed = any(
                i > repair_idx and s.stage == "verifier" and s.ok for i, s in enumerate(turn.stages)
            )
        if verifier_failed and verifier_repassed:
            failed = [stage for stage in failed if stage != "verifier"]
        if failed:
            return (
                False,
                f"family {turn.family} stage failed: {','.join(failed)}",
                {
                    "family": turn.family,
                    "failed": failed,
                },
            )
    if any(not turn.reply_signature for turn in telemetries):
        return (
            False,
            "missing reply signatures for diversity check",
            {
                "turns": len(telemetries),
                "families": sorted({turn.family for turn in telemetries}),
            },
        )
    signatures = [turn.reply_signature for turn in telemetries]
    ok, detail = diversity_passes(signatures)
    metrics: dict[str, Any] = {
        "turns": len(telemetries),
        "families": sorted(families),
        "stage_totals_ms": {
            key: round(value, 3) for key, value in stage_totals(telemetries).items()
        },
        "end_to_end_p50_ms": round(percentile_ms([t.total_ms() for t in telemetries], 50), 3),
        "end_to_end_p95_ms": round(percentile_ms([t.total_ms() for t in telemetries], 95), 3),
        "diversity": detail,
    }
    if not ok:
        return False, detail, metrics
    return True, "gate C telemetry covers stages with diverse replies", metrics


def evaluate_gate_e_telemetry(
    telemetries: list[TurnTelemetry],
    *,
    heartbeat_sends: int,
    heartbeat_interval_ms: float,
    p95_target_ms: float = 15000.0,
) -> tuple[bool, str, dict[str, Any]]:
    """Evaluate Gate E SLO telemetry: p50/p95, budget, heartbeat."""
    if not telemetries:
        return (
            False,
            "no live latency samples",
            {
                "turns": 0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "max_ms": 0.0,
                "budget_ms": float(ORDINARY_TURN_BUDGET_MS),
                "p95_target_ms": float(p95_target_ms),
                "heartbeat": "heartbeat never fired",
                "heartbeat_sends": int(heartbeat_sends),
            },
        )
    latencies = [turn.total_ms() for turn in telemetries]
    ok, detail, slo = slo_guards(latencies, p95_target_ms=p95_target_ms)
    duration = sum(latencies)
    hb_ok, hb_detail = heartbeat_continuity_ok(
        sends=heartbeat_sends,
        duration_ms=duration,
        interval_ms=heartbeat_interval_ms,
    )
    metrics: dict[str, Any] = {
        "turns": len(telemetries),
        "p50_ms": round(slo["p50_ms"], 3),
        "p95_ms": round(slo["p95_ms"], 3),
        "max_ms": round(slo["max_ms"], 3),
        "budget_ms": slo["budget_ms"],
        "p95_target_ms": slo["p95_target_ms"],
        "heartbeat": hb_detail,
        "heartbeat_sends": heartbeat_sends,
    }
    if not ok:
        return False, detail, metrics
    if not hb_ok:
        return False, hb_detail, metrics
    return True, "slo and heartbeat continuity within guard", metrics


__all__ = [
    "OPTIONAL_STAGES",
    "STAGES",
    "StageSample",
    "TurnTelemetry",
    "evaluate_gate_c_telemetry",
    "evaluate_gate_e_telemetry",
    "record_stage",
    "reply_signature",
    "stage_totals",
]
