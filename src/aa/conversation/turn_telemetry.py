"""Privacy-safe stage outcome + latency telemetry for the v2 turn path.

Every ordinary Telegram turn flows through::

    planner -> retrieval -> answer -> verifier -> [repair] -> delivery

The production regression in issue #145 collapsed all of those stages into
one generic clarification reply with no way to distinguish which stage
failed and no latency evidence. This module records per-stage outcomes and
wall-clock latencies using only counts, lengths, digests and categories:
never user text, prompts, model outputs, evidence text or raw identifiers.

The helpers are transport-independent so unit tests, the LangGraph
runtime and the Telegram delivery boundary can share one privacy-safe
vocabulary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def now_ms() -> float:
    """Return a monotonic millisecond clock for stage timing."""
    return time.perf_counter() * 1000.0


def percentile(values: list[float], pct: float) -> float:
    """Return the nearest-rank percentile of ``values`` (0 when empty)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[rank])


@dataclass(frozen=True)
class TurnTelemetry:
    """Privacy-safe outcome + latency snapshot for one ordinary turn.

    All fields are counts, lengths, latencies or fixed outcome tokens.
    No user text, prompt text, evidence text or raw chat identifier is
    stored here.
    """

    planner_outcome: str = "unknown"
    planner_latency_ms: float = 0.0
    planner_query_count: int = 0
    retrieval_outcome: str = "unknown"
    retrieval_latency_ms: float = 0.0
    retrieval_passages: int = 0
    retrieval_over_budget: bool = False
    answer_outcome: str = "unknown"
    answer_latency_ms: float = 0.0
    answer_rounds: int = 0
    verifier_outcome: str = "unknown"
    verifier_latency_ms: float = 0.0
    repair_rounds: int = 0
    delivery_outcome: str = "unknown"
    delivery_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    reply_len: int = 0
    envelope_ok: bool = True
    fallback_reply: bool = False

    def to_safe_dict(self) -> dict[str, object]:
        """Return a log/artifact-safe mapping (no text payloads)."""
        return {
            "planner_outcome": self.planner_outcome,
            "planner_latency_ms": round(float(self.planner_latency_ms), 1),
            "planner_query_count": int(self.planner_query_count),
            "retrieval_outcome": self.retrieval_outcome,
            "retrieval_latency_ms": round(float(self.retrieval_latency_ms), 1),
            "retrieval_passages": int(self.retrieval_passages),
            "retrieval_over_budget": bool(self.retrieval_over_budget),
            "answer_outcome": self.answer_outcome,
            "answer_latency_ms": round(float(self.answer_latency_ms), 1),
            "answer_rounds": int(self.answer_rounds),
            "verifier_outcome": self.verifier_outcome,
            "verifier_latency_ms": round(float(self.verifier_latency_ms), 1),
            "repair_rounds": int(self.repair_rounds),
            "delivery_outcome": self.delivery_outcome,
            "delivery_latency_ms": round(float(self.delivery_latency_ms), 1),
            "total_latency_ms": round(float(self.total_latency_ms), 1),
            "reply_len": int(self.reply_len),
            "envelope_ok": bool(self.envelope_ok),
            "fallback_reply": bool(self.fallback_reply),
        }


@dataclass
class TurnTimer:
    """Mutable per-turn stopwatch accumulating stage latencies."""

    _started_ms: float = field(default_factory=now_ms)
    stages: dict[str, float] = field(default_factory=dict)

    def start_stage(self, name: str) -> float:
        """Mark the start of ``name`` and return its start timestamp."""
        started = now_ms()
        self.stages[f"{name}_start"] = started
        return started

    def end_stage(self, name: str, started: float) -> float:
        """Record elapsed milliseconds for ``name`` since ``started``."""
        elapsed = max(0.0, now_ms() - started)
        self.stages[name] = elapsed
        return elapsed

    def stage_ms(self, name: str) -> float:
        """Return recorded milliseconds for ``name`` (0 when absent)."""
        value = self.stages.get(name, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0

    def total_ms(self) -> float:
        """Return milliseconds elapsed since this timer was created."""
        return max(0.0, now_ms() - self._started_ms)


def summarize_latencies(latencies_s: list[float]) -> dict[str, float]:
    """Return privacy-safe p50/p95/count summary for live text latency."""
    return {
        "count": float(len(latencies_s)),
        "p50_s": round(percentile(list(latencies_s), 50), 4),
        "p95_s": round(percentile(list(latencies_s), 95), 4),
        "max_s": round(max(latencies_s) if latencies_s else 0.0, 4),
    }


__all__ = ["TurnTelemetry", "TurnTimer", "now_ms", "percentile", "summarize_latencies"]
