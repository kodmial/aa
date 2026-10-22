"""Single validated time-policy seam (kodmial/aa#408).

Runtime time limits are disabled by default so slow-but-successful model
turns complete. Qualification keeps measuring every ordinary turn and Gate E
runs in ``observe`` mode by default: timing is published but the old
p95/max thresholds are not evaluated. Operators can restore the previous
behaviour without code changes via environment variables.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

OBSERVE_DETAIL = "Gate E timing observed — SLO NOT EVALUATED"
VALID_LATENCY_GATE_MODES = ("observe", "enforce")

DEFAULT_TURN_TIME_LIMITS_ENABLED = False
DEFAULT_LATENCY_GATE_MODE = "observe"
DEFAULT_LATENCY_P95_TARGET_MS = 60_000.0
DEFAULT_LATENCY_MAX_TURN_MS = 120_000.0
# Transport policy (kodmial/aa#412): ``transport_timeout_s`` is a
# connect/inactivity bound for one blocking socket operation, never an
# end-to-end response deadline. A healthy slow model may stay silent well
# past the former fixed walls; only the independently supervised
# qualification campaign deadline (with finalisation reserve) can end it.
DEFAULT_TRANSPORT_TIMEOUT_S = 1800.0
DEFAULT_TRANSPORT_CONNECT_TIMEOUT_S = 30.0
# One coherent 135-minute exact-main campaign. The parent watchdog subtracts
# the explicit finalisation reserve before using this allowance.
DEFAULT_QUALIFICATION_CAMPAIGN_S = 8100.0
DEFAULT_QUALIFICATION_FINALISATION_RESERVE_S = 900.0
# A scenario is stopped only when no *real* progress event (stage
# transition, delivered response, completed scenario) has been observed for
# this long. A polling heartbeat never resets this bound.
DEFAULT_QUALIFICATION_SCENARIO_INACTIVITY_S = 1800.0
# Legacy projection retained for evidence consumers; it is a campaign hint,
# never a per-scenario hard cutoff.
DEFAULT_QUALIFICATION_SCENARIO_WAIT_S = DEFAULT_QUALIFICATION_CAMPAIGN_S
DEFAULT_TURN_END_TO_END_BUDGET_S = 105.0
DEFAULT_ANSWER_DRAFT_ATTEMPT_BUDGET_S = 35.0
DEFAULT_REPAIR_TIME_BUDGET_S = 50.0
DEFAULT_PLANNER_TIME_BUDGET_S = 25.0
DEFAULT_PLANNER_STRUCTURED_ATTEMPT_BUDGET_S = 6.0
DEFAULT_RETRIEVAL_SEMANTIC_STAGE_BUDGET_S = 20.0
DEFAULT_RETRIEVAL_PER_CALL_BUDGET_S = 8.0
DEFAULT_VERIFIER_TURN_BUDGET_S = 40.0
DEFAULT_VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S = 6.0

# Environment names that participate in policy resolution. The cache key
# below includes their current values, so a process that changes env vars
# (tests, controlled reload) still sees the new policy immediately.
_TIME_POLICY_ENV_NAMES: tuple[str, ...] = (
    "AA_TURN_TIME_LIMITS_ENABLED",
    "AA_LATENCY_GATE_MODE",
    "AA_LATENCY_P95_TARGET_MS",
    "AA_LATENCY_MAX_TURN_MS",
    "AA_TRANSPORT_TIMEOUT_S",
    "AA_TRANSPORT_CONNECT_TIMEOUT_S",
    "AA_QUALIFICATION_CAMPAIGN_S",
    "AA_QUALIFICATION_JOB_TIMEOUT_MINUTES",
    "AA_QUALIFICATION_FINALISATION_RESERVE_S",
    "AA_QUALIFICATION_SCENARIO_WAIT_S",
    "AA_QUALIFICATION_SCENARIO_INACTIVITY_S",
    "AA_TURN_END_TO_END_BUDGET_S",
    "AA_ANSWER_DRAFT_ATTEMPT_BUDGET_S",
    "AA_REPAIR_TIME_BUDGET_S",
    "AA_PLANNER_TIME_BUDGET_S",
    "AA_PLANNER_STRUCTURED_ATTEMPT_BUDGET_S",
    "AA_RETRIEVAL_SEMANTIC_STAGE_BUDGET_S",
    "AA_RETRIEVAL_PER_CALL_BUDGET_S",
    "AA_VERIFIER_TURN_BUDGET_S",
    "AA_VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S",
)
_POLICY_CACHE_MAX = 16
_POLICY_CACHE: dict[tuple[str | None, ...], TimePolicy] = {}


class TimePolicyError(ValueError):
    """Raised when time-policy configuration is invalid."""


@dataclass(frozen=True)
class TimePolicy:
    """Validated runtime and qualification timing policy."""

    turn_time_limits_enabled: bool = DEFAULT_TURN_TIME_LIMITS_ENABLED
    latency_gate_mode: str = DEFAULT_LATENCY_GATE_MODE
    latency_p95_target_ms: float = DEFAULT_LATENCY_P95_TARGET_MS
    latency_max_turn_ms: float = DEFAULT_LATENCY_MAX_TURN_MS
    transport_timeout_s: float = DEFAULT_TRANSPORT_TIMEOUT_S
    transport_connect_timeout_s: float = DEFAULT_TRANSPORT_CONNECT_TIMEOUT_S
    qualification_campaign_s: float = DEFAULT_QUALIFICATION_CAMPAIGN_S
    qualification_finalisation_reserve_s: float = DEFAULT_QUALIFICATION_FINALISATION_RESERVE_S
    qualification_scenario_wait_s: float = DEFAULT_QUALIFICATION_SCENARIO_WAIT_S
    qualification_scenario_inactivity_s: float = DEFAULT_QUALIFICATION_SCENARIO_INACTIVITY_S
    turn_end_to_end_budget_s: float = DEFAULT_TURN_END_TO_END_BUDGET_S
    answer_draft_attempt_budget_s: float = DEFAULT_ANSWER_DRAFT_ATTEMPT_BUDGET_S
    repair_time_budget_s: float = DEFAULT_REPAIR_TIME_BUDGET_S
    planner_time_budget_s: float = DEFAULT_PLANNER_TIME_BUDGET_S
    planner_structured_attempt_budget_s: float = DEFAULT_PLANNER_STRUCTURED_ATTEMPT_BUDGET_S
    retrieval_semantic_stage_budget_s: float = DEFAULT_RETRIEVAL_SEMANTIC_STAGE_BUDGET_S
    retrieval_per_call_budget_s: float = DEFAULT_RETRIEVAL_PER_CALL_BUDGET_S
    verifier_turn_budget_s: float = DEFAULT_VERIFIER_TURN_BUDGET_S
    verifier_structured_attempt_budget_s: float = DEFAULT_VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S

    def __post_init__(self) -> None:
        if not isinstance(self.turn_time_limits_enabled, bool):
            raise TimePolicyError("turn_time_limits_enabled must be a boolean")
        if self.latency_gate_mode not in VALID_LATENCY_GATE_MODES:
            raise TimePolicyError(f"latency_gate_mode must be one of {VALID_LATENCY_GATE_MODES}")
        for name in (
            "latency_p95_target_ms",
            "latency_max_turn_ms",
            "transport_timeout_s",
            "transport_connect_timeout_s",
            "qualification_campaign_s",
            "qualification_finalisation_reserve_s",
            "qualification_scenario_wait_s",
            "qualification_scenario_inactivity_s",
            "turn_end_to_end_budget_s",
            "answer_draft_attempt_budget_s",
            "repair_time_budget_s",
            "planner_time_budget_s",
            "planner_structured_attempt_budget_s",
            "retrieval_semantic_stage_budget_s",
            "retrieval_per_call_budget_s",
            "verifier_turn_budget_s",
            "verifier_structured_attempt_budget_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TimePolicyError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise TimePolicyError(f"{name} must be finite positive")
        if self.latency_max_turn_ms < self.latency_p95_target_ms:
            raise TimePolicyError("latency_max_turn_ms must be >= latency_p95_target_ms")
        if self.qualification_finalisation_reserve_s >= self.qualification_campaign_s:
            raise TimePolicyError(
                "qualification_finalisation_reserve_s must be smaller than qualification_campaign_s"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> TimePolicy:
        """Resolve and validate the policy from environment variables."""
        source = os.environ if environ is None else environ
        return cls(
            turn_time_limits_enabled=_env_bool(
                source, "AA_TURN_TIME_LIMITS_ENABLED", DEFAULT_TURN_TIME_LIMITS_ENABLED
            ),
            latency_gate_mode=_env_str(
                source, "AA_LATENCY_GATE_MODE", DEFAULT_LATENCY_GATE_MODE
            ).lower(),
            latency_p95_target_ms=_env_float(
                source, "AA_LATENCY_P95_TARGET_MS", DEFAULT_LATENCY_P95_TARGET_MS
            ),
            latency_max_turn_ms=_env_float(
                source, "AA_LATENCY_MAX_TURN_MS", DEFAULT_LATENCY_MAX_TURN_MS
            ),
            transport_timeout_s=_env_float(
                source, "AA_TRANSPORT_TIMEOUT_S", DEFAULT_TRANSPORT_TIMEOUT_S
            ),
            transport_connect_timeout_s=_env_float(
                source, "AA_TRANSPORT_CONNECT_TIMEOUT_S", DEFAULT_TRANSPORT_CONNECT_TIMEOUT_S
            ),
            qualification_campaign_s=_env_float(
                source, "AA_QUALIFICATION_CAMPAIGN_S", DEFAULT_QUALIFICATION_CAMPAIGN_S
            ),
            qualification_finalisation_reserve_s=_env_float(
                source,
                "AA_QUALIFICATION_FINALISATION_RESERVE_S",
                DEFAULT_QUALIFICATION_FINALISATION_RESERVE_S,
            ),
            qualification_scenario_wait_s=_env_float(
                source,
                "AA_QUALIFICATION_SCENARIO_WAIT_S",
                DEFAULT_QUALIFICATION_SCENARIO_WAIT_S,
            ),
            qualification_scenario_inactivity_s=_env_float(
                source,
                "AA_QUALIFICATION_SCENARIO_INACTIVITY_S",
                DEFAULT_QUALIFICATION_SCENARIO_INACTIVITY_S,
            ),
            turn_end_to_end_budget_s=_env_float(
                source, "AA_TURN_END_TO_END_BUDGET_S", DEFAULT_TURN_END_TO_END_BUDGET_S
            ),
            answer_draft_attempt_budget_s=_env_float(
                source,
                "AA_ANSWER_DRAFT_ATTEMPT_BUDGET_S",
                DEFAULT_ANSWER_DRAFT_ATTEMPT_BUDGET_S,
            ),
            repair_time_budget_s=_env_float(
                source, "AA_REPAIR_TIME_BUDGET_S", DEFAULT_REPAIR_TIME_BUDGET_S
            ),
            planner_time_budget_s=_env_float(
                source, "AA_PLANNER_TIME_BUDGET_S", DEFAULT_PLANNER_TIME_BUDGET_S
            ),
            planner_structured_attempt_budget_s=_env_float(
                source,
                "AA_PLANNER_STRUCTURED_ATTEMPT_BUDGET_S",
                DEFAULT_PLANNER_STRUCTURED_ATTEMPT_BUDGET_S,
            ),
            retrieval_semantic_stage_budget_s=_env_float(
                source,
                "AA_RETRIEVAL_SEMANTIC_STAGE_BUDGET_S",
                DEFAULT_RETRIEVAL_SEMANTIC_STAGE_BUDGET_S,
            ),
            retrieval_per_call_budget_s=_env_float(
                source, "AA_RETRIEVAL_PER_CALL_BUDGET_S", DEFAULT_RETRIEVAL_PER_CALL_BUDGET_S
            ),
            verifier_turn_budget_s=_env_float(
                source, "AA_VERIFIER_TURN_BUDGET_S", DEFAULT_VERIFIER_TURN_BUDGET_S
            ),
            verifier_structured_attempt_budget_s=_env_float(
                source,
                "AA_VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S",
                DEFAULT_VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S,
            ),
        )

    def with_latency_mode(self, mode: str) -> TimePolicy:
        """Return a copy with an explicit Gate E mode (validated)."""
        return replace(self, latency_gate_mode=str(mode).strip().lower())

    def to_safe_dict(self) -> dict[str, object]:
        """Return a privacy-safe policy projection for config/evidence."""
        return {
            "turn_time_limits_enabled": self.turn_time_limits_enabled,
            "latency_gate_mode": self.latency_gate_mode,
            "latency_p95_target_ms": self.latency_p95_target_ms,
            "latency_max_turn_ms": self.latency_max_turn_ms,
            "transport_timeout_s": self.transport_timeout_s,
            "transport_connect_timeout_s": self.transport_connect_timeout_s,
            "qualification_campaign_s": self.qualification_campaign_s,
            "qualification_finalisation_reserve_s": self.qualification_finalisation_reserve_s,
            "qualification_scenario_wait_s": self.qualification_scenario_wait_s,
            "qualification_scenario_inactivity_s": self.qualification_scenario_inactivity_s,
        }


def current_time_policy(environ: Mapping[str, str] | None = None) -> TimePolicy:
    """Resolve the policy for the current process environment.

    Explicit mappings are always parsed as given. For the live process
    environment, validated policies are cached by the values of the
    participating variables, avoiding repeated object construction in
    per-turn hot paths without hiding env changes.
    """
    if environ is not None:
        return TimePolicy.from_env(environ)
    key = tuple(os.environ.get(name) for name in _TIME_POLICY_ENV_NAMES)
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    policy = TimePolicy.from_env()
    if len(_POLICY_CACHE) >= _POLICY_CACHE_MAX:
        _POLICY_CACHE.clear()
    _POLICY_CACHE[key] = policy
    return policy


def evaluate_latency_thresholds(
    p95_ms: float,
    max_turn_ms: float,
    *,
    policy: TimePolicy | None = None,
) -> tuple[bool, str]:
    """Evaluate configured Gate E thresholds for one latency aggregate.

    In ``observe`` mode timing is never a FAIL; the caller still owns
    reliability/evidence-integrity failures such as censored timeouts.
    """
    active = policy if policy is not None else current_time_policy()
    if active.latency_gate_mode == "observe":
        return True, OBSERVE_DETAIL
    if max_turn_ms >= active.latency_max_turn_ms:
        return (
            False,
            (
                f"ordinary turn took {max_turn_ms:.0f}ms >= "
                f"{active.latency_max_turn_ms:.0f}ms budget"
            ),
        )
    if p95_ms > active.latency_p95_target_ms:
        return (
            False,
            (f"p95 {p95_ms:.0f}ms exceeds target {active.latency_p95_target_ms:.0f}ms"),
        )
    return True, "slo within guard"


def _env_str(source: Mapping[str, str], name: str, default: str) -> str:
    raw = source.get(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


def _env_bool(source: Mapping[str, str], name: str, default: bool) -> bool:
    raw = source.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise TimePolicyError(f"{name} must be a boolean (true/false)")


def _env_float(source: Mapping[str, str], name: str, default: float) -> float:
    raw = source.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise TimePolicyError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise TimePolicyError(f"{name} must be finite positive")
    return value


__all__ = [
    "DEFAULT_ANSWER_DRAFT_ATTEMPT_BUDGET_S",
    "DEFAULT_LATENCY_GATE_MODE",
    "DEFAULT_LATENCY_MAX_TURN_MS",
    "DEFAULT_LATENCY_P95_TARGET_MS",
    "DEFAULT_QUALIFICATION_CAMPAIGN_S",
    "DEFAULT_QUALIFICATION_FINALISATION_RESERVE_S",
    "DEFAULT_QUALIFICATION_SCENARIO_INACTIVITY_S",
    "DEFAULT_PLANNER_STRUCTURED_ATTEMPT_BUDGET_S",
    "DEFAULT_PLANNER_TIME_BUDGET_S",
    "DEFAULT_QUALIFICATION_SCENARIO_WAIT_S",
    "DEFAULT_REPAIR_TIME_BUDGET_S",
    "DEFAULT_RETRIEVAL_PER_CALL_BUDGET_S",
    "DEFAULT_RETRIEVAL_SEMANTIC_STAGE_BUDGET_S",
    "DEFAULT_TRANSPORT_CONNECT_TIMEOUT_S",
    "DEFAULT_TRANSPORT_TIMEOUT_S",
    "DEFAULT_TURN_END_TO_END_BUDGET_S",
    "DEFAULT_TURN_TIME_LIMITS_ENABLED",
    "DEFAULT_VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S",
    "DEFAULT_VERIFIER_TURN_BUDGET_S",
    "OBSERVE_DETAIL",
    "VALID_LATENCY_GATE_MODES",
    "TimePolicy",
    "TimePolicyError",
    "current_time_policy",
    "evaluate_latency_thresholds",
]
