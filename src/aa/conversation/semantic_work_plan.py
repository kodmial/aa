"""Single coherent budget-aware semantic work plan (kodmial/aa#380).

Replaces the multiplicative sequential LLM workload with one explicit
per-turn plan plus a privacy-safe stage critical-path DAG on the real
compiled LangGraph path.

Workload contract enforced here:

- exactly one verifier round per draft, always in
  ``batched-single-round`` mode: one model call carries every response
  unit with the Evidence Pack materialized once, and returns one
  independently explicit verdict per unit (strict completeness, exact
  citations, integrity/quote/provenance and intent-relevance
  validation downstream);
- bounded full replan/retrieve repair rounds per turn, each gated
  on genuine semantic progress (new passages or newly supported
  units): the full pipeline never repeats on no-progress, and
  same-pack regeneration from the immutable Evidence Pack is
  preferred over another retrieval round;
- every provider call is deadline-aware before it is issued: the plan
  reserves slices for the generator, the grounding verifier, the
  independent whole-turn judge and finalization, so no turn waits past
  the internal budget into the hard SLO.

Privacy: the DAG stores only stage names, boolean outcomes, latencies,
counts, token estimates, model identities and short digests. It never
stores user text, book text, prompts, replies, summaries, chat ids or
secrets. This module is intentionally leaf-level (stdlib only) so every
graph stage can record without import cycles.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Internal end-to-end turn budget in seconds. Must stay equal to
# ``aa.conversation.turn_pipeline.TURN_END_TO_END_BUDGET_S`` and
# ``aa.conversation.turn_budget.TURN_END_TO_END_BUDGET_S`` (105s with
# delivery margin under the hard SLO); enforced by tests.
INTERNAL_TURN_BUDGET_S = 105.0

# Authoritative unchanged Gate E hard SLO: no ordinary turn may await
# provider work past 120s. Must stay equal to
# ``aa.conversation.turn_budget.TURN_HARD_SLO_MS``; enforced by tests.
HARD_TURN_SLO_S = 120.0

# Authoritative unchanged Gate E p95 target (60s). Duplicated so this
# leaf module never imports the qualification DAG; enforced by tests.
GATE_E_P95_TARGET_S = 60.0

# Minimum useful slices of the remaining budget. Below the answer slice
# no answer call starts; below the verifier slice no verifier round
# starts; the judge slice caps one judge round after reserving the
# verifier slice; finalization keeps a small local margin. Values match
# the turn-pipeline slices so both layers agree; enforced by tests.
ANSWER_MIN_SLICE_S = 1.0
VERIFIER_MIN_SLICE_S = 3.0
JUDGE_CALL_CAP_S = 20.0
FINALIZE_RESERVE_S = 2.0

# Single coherent verifier topology: one batched round per draft.
VERIFIER_ROUND_MODE = "batched-single-round"

# Bounded full replan/retrieve repair rounds per turn (matches
# ``aa.conversation.turn_pipeline.MAX_TARGETED_REPAIR_ROUNDS``).
# Every round past the initial attempt additionally requires measured
# semantic progress (new canonical evidence or newly supported units);
# a round with no progress stops the loop instead of repeating the
# full pipeline. Same-pack regen from the immutable Evidence Pack is
# always preferred over another retrieval round.
MAX_FULL_REPAIR_ROUNDS = 2


def new_turn_dag_id() -> str:
    """Return a fresh privacy-safe per-turn DAG id (hashed, no text)."""
    raw = f"{uuid.uuid4().hex}:{time.monotonic_ns()}:{threading.get_ident()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def digest_bytes(payload: bytes) -> str:
    """Return the hex digest identifying one exact byte string.

    Used only as an exact-equality cache key (for example reusing
    validated verifier verdicts for a byte-identical draft against a
    digest-identical Evidence Pack within one turn). It never selects
    reply variety: different bytes always miss.
    """
    try:
        return hashlib.sha256(bytes(payload or b"")).hexdigest()
    except Exception:
        return ""


def estimate_tokens_for_chars(char_count: int) -> int:
    """Estimate model tokens for a character count (accounting only)."""
    try:
        count = int(char_count)
    except (TypeError, ValueError):
        return 0
    if count <= 0:
        return 0
    try:
        from aa.corpus.budget import estimate_text_tokens as _estimate

        return int(_estimate("x" * min(count, 4096)) * (count / min(count, 4096)))
    except Exception:
        return max(1, count // 4)


@dataclass
class _OpRecord:
    """One critical-path operation (in-flight until finished)."""

    name: str
    started_monotonic: float = field(default_factory=time.perf_counter)
    finished_monotonic: float = 0.0
    ok: bool = False
    category: str = ""
    attempts: int = 0
    timeouts: int = 0
    cancellations: int = 0
    rate_limits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def latency_ms(self) -> float:
        end = self.finished_monotonic or time.perf_counter()
        return max(0.0, (end - self.started_monotonic) * 1000.0)


class SemanticTurnDAG:
    """Privacy-safe per-turn critical-path DAG (counts/digests only)."""

    def __init__(self, *, dag_id: str | None = None) -> None:
        self.dag_id: str = dag_id or new_turn_dag_id()
        self.started_monotonic: float = time.perf_counter()
        self._lock = threading.Lock()
        self._ops: dict[str, _OpRecord] = {}
        self._order: list[str] = []
        self._selection_iterations = 0
        self._coverage_iterations = 0
        self._coverage_expansions = 0
        self._coverage_searches = 0
        self._verifier_units = 0
        self._verifier_evidence_chars = 0
        self._verifier_materialized_chars = 0
        self._judge_rounds = 0

    def start_op(self, name: str) -> None:
        """Mark one semantic operation in-flight (never raises)."""
        try:
            clean = str(name or "").strip()[:64] or "unknown"
            with self._lock:
                if clean not in self._ops:
                    self._ops[clean] = _OpRecord(name=clean)
                    self._order.append(clean)
        except Exception:
            pass

    def finish_op(self, name: str, *, ok: bool, category: str = "") -> None:
        """Mark one semantic operation done (never raises)."""
        try:
            clean = str(name or "").strip()[:64] or "unknown"
            with self._lock:
                record = self._ops.get(clean)
                if record is None:
                    record = _OpRecord(name=clean)
                    self._ops[clean] = record
                    self._order.append(clean)
                record.finished_monotonic = time.perf_counter()
                record.ok = bool(ok)
                record.category = str(category or "")[:64]
        except Exception:
            pass

    def _bump(self, name: str, field_name: str, *, input_tokens: int = 0) -> None:
        try:
            clean = str(name or "").strip()[:64] or "unknown"
            with self._lock:
                record = self._ops.get(clean)
                if record is None:
                    record = _OpRecord(name=clean)
                    self._ops[clean] = record
                    self._order.append(clean)
                setattr(record, field_name, int(getattr(record, field_name, 0)) + 1)
                if input_tokens > 0:
                    record.input_tokens += int(input_tokens)
        except Exception:
            pass

    def provider_attempt(self, name: str, *, input_tokens: int = 0) -> None:
        """Record one issued provider call (never raises)."""
        self._bump(name, "attempts", input_tokens=input_tokens)

    def provider_timeout(self, name: str) -> None:
        """Record one provider-call timeout/censorship event (never raises)."""
        self._bump(name, "timeouts")

    def provider_cancelled(self, name: str) -> None:
        """Record one provider-call cancellation (never raises)."""
        self._bump(name, "cancellations")

    def provider_rate_limited(self, name: str) -> None:
        """Record one provider 429 (never raises; 429 still propagates)."""
        self._bump(name, "rate_limits")

    def record_token_usage(
        self, name: str, *, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """Record estimated model/token usage for one op (never raises)."""
        try:
            clean = str(name or "").strip()[:64] or "unknown"
            with self._lock:
                record = self._ops.get(clean)
                if record is None:
                    record = _OpRecord(name=clean)
                    self._ops[clean] = record
                    self._order.append(clean)
                record.input_tokens += max(0, int(input_tokens))
                record.output_tokens += max(0, int(output_tokens))
        except Exception:
            pass

    def record_selection_iteration(self, *, candidates: int, selected: int) -> None:
        """Record one retrieval selection iteration (never raises)."""
        try:
            with self._lock:
                self._selection_iterations += 1
        except Exception:
            pass

    def record_coverage_iteration(
        self, *, covered: int, uncovered: int, expansions: int = 0, searches: int = 0
    ) -> None:
        """Record one coverage assessment iteration (never raises)."""
        try:
            with self._lock:
                self._coverage_iterations += 1
                self._coverage_expansions += max(0, int(expansions))
                self._coverage_searches += max(0, int(searches))
        except Exception:
            pass

    def record_verifier_round(
        self, *, unit_count: int, evidence_chars: int, materialized_chars: int
    ) -> None:
        """Record one verifier round: unit count and evidence sizes."""
        try:
            with self._lock:
                self._verifier_units += max(0, int(unit_count))
                self._verifier_evidence_chars += max(0, int(evidence_chars))
                self._verifier_materialized_chars += max(0, int(materialized_chars))
        except Exception:
            pass

    def record_judge_round(self) -> None:
        """Record one whole-turn judge round (never raises)."""
        try:
            with self._lock:
                self._judge_rounds += 1
        except Exception:
            pass

    def in_flight_ops(self) -> list[str]:
        """Return names of operations started but not finished."""
        try:
            with self._lock:
                return [
                    record.name
                    for record in (self._ops[name] for name in self._order)
                    if not record.finished_monotonic
                ]
        except Exception:
            return []

    def elapsed_ms(self) -> float:
        return max(0.0, (time.perf_counter() - self.started_monotonic) * 1000.0)

    def snapshot(self) -> dict[str, Any]:
        """Return the privacy-safe DAG snapshot (no text payloads)."""
        try:
            with self._lock:
                ops = [
                    {
                        "op": record.name,
                        "ok": bool(record.ok),
                        "latency_ms": round(record.latency_ms(), 3),
                        "category": record.category,
                        "attempts": int(record.attempts),
                        "timeouts": int(record.timeouts),
                        "cancellations": int(record.cancellations),
                        "rate_limits": int(record.rate_limits),
                        "input_tokens": int(record.input_tokens),
                        "output_tokens": int(record.output_tokens),
                        "done": bool(record.finished_monotonic),
                    }
                    for record in (self._ops[name] for name in self._order)
                ]
                payload: dict[str, Any] = {
                    "dag_id": self.dag_id,
                    "elapsed_ms": round(self.elapsed_ms(), 3),
                    "ops": ops,
                    "selection_iterations": int(self._selection_iterations),
                    "coverage_iterations": int(self._coverage_iterations),
                    "coverage_expansions": int(self._coverage_expansions),
                    "coverage_searches": int(self._coverage_searches),
                    "verifier_units": int(self._verifier_units),
                    "verifier_evidence_chars": int(self._verifier_evidence_chars),
                    "verifier_materialized_chars": int(self._verifier_materialized_chars),
                    "judge_rounds": int(self._judge_rounds),
                    "in_flight": [
                        name for name in self._order if not self._ops[name].finished_monotonic
                    ],
                }
            return payload
        except Exception:
            return {"dag_id": self.dag_id, "ops": []}


_DAG_REGISTRY: dict[str, SemanticTurnDAG] = {}
_DAG_LOCK = threading.Lock()
_DAG_REGISTRY_MAX = 256

_DAG_STATE_KEY = "semantic_dag_id"


def register_dag(dag: SemanticTurnDAG) -> SemanticTurnDAG:
    """Register one DAG for graph-state sharing (bounded registry)."""
    try:
        with _DAG_LOCK:
            while len(_DAG_REGISTRY) >= _DAG_REGISTRY_MAX:
                _DAG_REGISTRY.pop(next(iter(_DAG_REGISTRY)))
            _DAG_REGISTRY[dag.dag_id] = dag
    except Exception:
        pass
    return dag


def dag_for_state(state: Any) -> SemanticTurnDAG | None:
    """Return the DAG attached to graph state, if any (never raises)."""
    try:
        if not isinstance(state, dict):
            return None
        dag_id = str(state.get(_DAG_STATE_KEY, "") or "").strip()
        if not dag_id:
            return None
        with _DAG_LOCK:
            return _DAG_REGISTRY.get(dag_id)
    except Exception:
        return None


def ensure_turn_dag(state: Any) -> SemanticTurnDAG:
    """Return the state DAG, creating and attaching one if absent."""
    try:
        existing = dag_for_state(state)
        if existing is not None:
            return existing
        created = register_dag(SemanticTurnDAG())
        if isinstance(state, dict):
            try:
                state[_DAG_STATE_KEY] = created.dag_id
            except Exception:
                pass
        return created
    except Exception:
        return SemanticTurnDAG()


def clear_dag_registry() -> None:
    """Drop all registered DAGs (tests/tooling only)."""
    try:
        with _DAG_LOCK:
            _DAG_REGISTRY.clear()
    except Exception:
        pass


def _policy_time_limits_enabled() -> bool:
    """Resolve the shared runtime time-policy switch (never raises)."""
    try:
        from aa.time_policy import current_time_policy

        return bool(current_time_policy().turn_time_limits_enabled)
    except Exception:
        return True


@dataclass(frozen=True)
class SemanticWorkPlan:
    """One coherent budget-aware semantic work plan for a turn.

    Computed once per turn from the remaining end-to-end budget. Every
    provider call is checked with :meth:`allows_provider_call` BEFORE it
    is issued, with slices reserved for the generator, the grounding
    verifier, the independent whole-turn judge and finalization, so a
    turn fails fast with a typed outcome instead of grinding past the
    internal budget into the hard SLO.
    """

    dag_id: str = ""
    internal_budget_s: float = INTERNAL_TURN_BUDGET_S
    hard_slo_s: float = HARD_TURN_SLO_S
    answer_min_slice_s: float = ANSWER_MIN_SLICE_S
    verifier_min_slice_s: float = VERIFIER_MIN_SLICE_S
    judge_call_cap_s: float = JUDGE_CALL_CAP_S
    finalize_reserve_s: float = FINALIZE_RESERVE_S
    verifier_round_mode: str = VERIFIER_ROUND_MODE
    max_full_repair_rounds: int = MAX_FULL_REPAIR_ROUNDS
    time_limits_enabled: bool | None = None

    def __post_init__(self) -> None:
        enabled = (
            _policy_time_limits_enabled()
            if self.time_limits_enabled is None
            else bool(self.time_limits_enabled)
        )
        object.__setattr__(self, "time_limits_enabled", enabled)

    def allows_provider_call(
        self, *, remaining_s: float, need_s: float, reserve_for: str = ""
    ) -> bool:
        """Whether a provider call may start (deadline-aware, fail-fast).

        ``remaining_s`` is the remaining end-to-end budget,
        ``need_s`` the minimum useful slice for the call, and
        ``reserve_for`` the downstream stage that must keep its slice
        (``"verifier"``, ``"judge"``, ``"finalize"`` or ``""``).

        When turn time limits are disabled by policy, no elapsed-time or
        downstream-reserve gate applies: finite count/progress limits are
        enforced elsewhere and remain active.
        """
        if not self.time_limits_enabled:
            return True
        try:
            remaining = float(remaining_s)
            need = float(need_s)
        except (TypeError, ValueError):
            return False
        if not remaining > 0 or not need > 0:
            return False
        reserve = 0.0
        try:
            token = str(reserve_for or "").strip().lower()
            if token == "verifier":
                reserve = float(self.verifier_min_slice_s)
            elif token == "judge":
                reserve = float(self.verifier_min_slice_s)
            elif token == "finalize":
                reserve = float(self.finalize_reserve_s)
        except (TypeError, ValueError):
            reserve = 0.0
        return bool(remaining >= need + reserve)

    def verifier_slice(self, remaining_s: float) -> float | None:
        """Remaining slice for one verifier round (``None`` = full default)."""
        if not self.time_limits_enabled:
            return None
        try:
            remaining = float(remaining_s)
        except (TypeError, ValueError):
            return 0.0
        try:
            from aa.time_policy import current_time_policy

            default = float(current_time_policy().verifier_turn_budget_s)
        except Exception:
            try:
                from aa.conversation.verifier import VERIFIER_TURN_BUDGET_S as default
            except Exception:
                default = 40.0
        if remaining >= float(default):
            return None
        return max(0.0, remaining)

    def judge_slice(self, remaining_s: float) -> float:
        """Capped slice for one judge round (``0`` = skip, budget-neutral)."""
        if not self.time_limits_enabled:
            return float("inf")
        try:
            remaining = float(remaining_s)
        except (TypeError, ValueError):
            return 0.0
        try:
            reserve = max(0.0, float(self.verifier_min_slice_s))
        except (TypeError, ValueError):
            reserve = VERIFIER_MIN_SLICE_S
        try:
            cap = min(max(0.0, remaining - reserve), float(self.judge_call_cap_s))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, float(cap))

    def repair_full_round_allowed(self, *, full_rounds_used: int) -> bool:
        """Whether another full replan/retrieve round may start."""
        try:
            return int(full_rounds_used) < int(self.max_full_repair_rounds)
        except (TypeError, ValueError):
            return False

    def describe(self) -> dict[str, Any]:
        """Return the privacy-safe plan description (no text payloads)."""
        return {
            "dag_id": str(self.dag_id or ""),
            "internal_budget_s": float(self.internal_budget_s),
            "hard_slo_s": float(self.hard_slo_s),
            "answer_min_slice_s": float(self.answer_min_slice_s),
            "verifier_min_slice_s": float(self.verifier_min_slice_s),
            "judge_call_cap_s": float(self.judge_call_cap_s),
            "finalize_reserve_s": float(self.finalize_reserve_s),
            "verifier_round_mode": str(self.verifier_round_mode or ""),
            "max_full_repair_rounds": int(self.max_full_repair_rounds),
            "time_limits_enabled": bool(self.time_limits_enabled),
        }


def new_work_plan(
    dag: SemanticTurnDAG | None = None,
    *,
    time_limits_enabled: bool | None = None,
) -> SemanticWorkPlan:
    """Build the per-turn work plan bound to one DAG (never raises).

    The plan follows the shared runtime time policy: when turn time
    limits are disabled, deadline/slice gates become inactive while all
    finite count/progress limits remain unchanged.
    """
    try:
        dag_id = str(dag.dag_id) if dag is not None else ""
    except Exception:
        dag_id = ""
    return SemanticWorkPlan(dag_id=dag_id, time_limits_enabled=time_limits_enabled)


@dataclass(frozen=True)
class RepairProgress:
    """Privacy-safe repair progress marker (counts only, never text)."""

    pack_passages: int = 0
    supported_units: int = 0


def repair_made_progress(before: RepairProgress, after: RepairProgress) -> bool:
    """Whether a repair round made genuine semantic progress.

    Progress is new canonical evidence (grown pack) or newly supported
    response units. Anything else is no-progress: the full pipeline
    must not repeat.
    """
    try:
        if int(after.pack_passages) > int(before.pack_passages):
            return True
        return int(after.supported_units) > int(before.supported_units)
    except (TypeError, ValueError):
        return False


def remaining_turn_s(
    *,
    turn_started_monotonic: float,
    upstream_spent_s: float = 0.0,
    budget_s: float = INTERNAL_TURN_BUDGET_S,
    time_limits_enabled: bool | None = None,
) -> float:
    """Return remaining end-to-end budget seconds (never raises).

    When turn time limits are disabled by policy the remaining budget is
    unbounded (``float("inf")``), so callers never turn elapsed time into
    a skip gate while finite count/progress controls stay active.
    """
    enabled = (
        _policy_time_limits_enabled() if time_limits_enabled is None else bool(time_limits_enabled)
    )
    if not enabled:
        return float("inf")
    try:
        elapsed = max(0.0, time.perf_counter() - float(turn_started_monotonic))
        upstream = max(0.0, float(upstream_spent_s or 0.0))
        return max(0.0, float(budget_s) - elapsed - upstream)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "ANSWER_MIN_SLICE_S",
    "FINALIZE_RESERVE_S",
    "GATE_E_P95_TARGET_S",
    "HARD_TURN_SLO_S",
    "INTERNAL_TURN_BUDGET_S",
    "JUDGE_CALL_CAP_S",
    "MAX_FULL_REPAIR_ROUNDS",
    "VERIFIER_MIN_SLICE_S",
    "VERIFIER_ROUND_MODE",
    "RepairProgress",
    "SemanticTurnDAG",
    "SemanticWorkPlan",
    "clear_dag_registry",
    "dag_for_state",
    "digest_bytes",
    "ensure_turn_dag",
    "estimate_tokens_for_chars",
    "new_turn_dag_id",
    "new_work_plan",
    "remaining_turn_s",
    "repair_made_progress",
    "register_dag",
]
