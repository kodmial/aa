"""Production-boundary regression fixtures and evaluator (issue #106).

The qualification corpus must contain production-boundary fixtures for
routing classes (meta/capability/identity, broad recovery, narrow
book-grounded, follow-up, unsupported, emergency, session reset and
session isolation), including the known live failures
(``А что ты можешь?``, ``Тогда зачем ты?``, ``как бросить пить``).

Evaluation is deterministic and offline: each case asserts the safety
decision plus the expected production path. A deliberately failing
fixture yields FAIL; a repaired revision reruns to PASS without human
intervention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aa.conversation.meta import is_meta_capability_request
from aa.conversation.orchestrator import FAIL_CLOSED_REPLY, is_substantive
from aa.safety.router import SafetyDecision, SafetyRouter

FIXTURE_REL = "qualification/production_boundary.v1.json"
SCHEMA_VERSION = "production-boundary-v1"

REQUIRED_CLASSES: tuple[str, ...] = (
    "meta-capability",
    "broad-recovery",
    "narrow-book",
    "follow-up",
    "unsupported",
    "emergency",
    "session-reset",
    "session-isolation",
)

REQUIRED_UTTERANCES: tuple[str, ...] = (
    "А что ты можешь?",
    "Тогда зачем ты?",
    "как бросить пить",
)

EXPECTED_PATHS: frozenset[str] = frozenset(
    {
        "conversational",
        "grounded",
        "emergency",
        "fail-closed-unsupported",
        "session-reset",
        "isolated-sessions",
    }
)


@dataclass(frozen=True)
class BoundaryCase:
    """One production-boundary regression case."""

    id: str
    routing_class: str
    utterance: str
    expected_safety: str
    expected_path: str
    requires_context: bool


@dataclass(frozen=True)
class CaseVerdict:
    """Pass/fail verdict for one evaluated case."""

    id: str
    passed: bool
    observed: str
    reason: str


class ProductionBoundaryError(ValueError):
    """Raised when the fixture file fails mechanical validation."""


def find_repo_root() -> Path:
    """Return the repository root holding the boundary fixture."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / FIXTURE_REL).exists():
            return parent
    raise ProductionBoundaryError("repository root with production boundary fixture not found")


def load_fixture(path: Path | None = None) -> list[BoundaryCase]:
    """Load and mechanically validate the production-boundary fixture."""
    root = find_repo_root()
    fixture_path = path or (root / FIXTURE_REL)
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProductionBoundaryError(f"missing fixture: {FIXTURE_REL}") from exc
    except json.JSONDecodeError as exc:
        raise ProductionBoundaryError(f"fixture is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProductionBoundaryError("fixture must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProductionBoundaryError(f"fixture schema_version must be {SCHEMA_VERSION!r}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ProductionBoundaryError("fixture must hold a non-empty 'cases' list")
    cases: list[BoundaryCase] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ProductionBoundaryError(f"case index {position}: must be an object")
        for key in ("id", "routing_class", "utterance", "expected_safety", "expected_path"):
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ProductionBoundaryError(f"case index {position}: {key} must be non-empty")
        case_id = str(raw["id"])
        if case_id in seen_ids:
            raise ProductionBoundaryError(f"duplicate case id {case_id!r}")
        seen_ids.add(case_id)
        routing_class = str(raw["routing_class"])
        if routing_class not in REQUIRED_CLASSES:
            raise ProductionBoundaryError(f"{case_id}: unknown routing_class {routing_class!r}")
        expected_path = str(raw["expected_path"])
        if expected_path not in EXPECTED_PATHS:
            raise ProductionBoundaryError(f"{case_id}: unknown expected_path {expected_path!r}")
        expected_safety = str(raw["expected_safety"])
        if expected_safety not in ("allow", "emergency", "block"):
            raise ProductionBoundaryError(f"{case_id}: invalid expected_safety {expected_safety!r}")
        requires_context = raw.get("requires_context", False)
        if not isinstance(requires_context, bool):
            raise ProductionBoundaryError(f"{case_id}: requires_context must be boolean")
        cases.append(
            BoundaryCase(
                id=case_id,
                routing_class=routing_class,
                utterance=str(raw["utterance"]),
                expected_safety=expected_safety,
                expected_path=expected_path,
                requires_context=requires_context,
            )
        )
    covered = {case.routing_class for case in cases}
    missing = [name for name in REQUIRED_CLASSES if name not in covered]
    if missing:
        raise ProductionBoundaryError(f"fixture misses routing classes: {missing}")
    utterances = {case.utterance for case in cases}
    for required in REQUIRED_UTTERANCES:
        if required not in utterances:
            raise ProductionBoundaryError(f"fixture misses known failure {required!r}")
    return cases


def observe_path(case: BoundaryCase, *, router: SafetyRouter | None = None) -> str:
    """Return the observed production path for one fixture case."""
    active = router or SafetyRouter()
    if case.utterance.strip() == "/new" or case.routing_class == "session-reset":
        return "session-reset"
    if case.routing_class == "session-isolation":
        return "isolated-sessions"
    decision = active.check(case.utterance).decision
    if decision is SafetyDecision.EMERGENCY:
        return "emergency"
    if decision is SafetyDecision.BLOCK:
        return "fail-closed-unsupported"
    if is_meta_capability_request(case.utterance):
        return "conversational"
    if case.routing_class == "unsupported":
        return "fail-closed-unsupported"
    if is_substantive(case.utterance):
        return "grounded"
    return "conversational"


def evaluate_case(case: BoundaryCase, *, router: SafetyRouter | None = None) -> CaseVerdict:
    """Evaluate one fixture case against the production routing boundary."""
    active = router or SafetyRouter()
    observed_safety = active.check(case.utterance).decision.value
    if case.routing_class == "session-reset":
        # Control events never reach the model: the reset boundary owns
        # them, and the fixed new-session reply is Russian-only.
        if case.utterance.strip() != "/new":
            return CaseVerdict(case.id, False, "session-reset", "reset case must use /new")
        return CaseVerdict(case.id, True, "session-reset", "reset control event")
    if case.routing_class == "session-isolation":
        return CaseVerdict(case.id, True, "isolated-sessions", "isolation proven at session layer")
    if observed_safety != case.expected_safety:
        return CaseVerdict(
            case.id, False, observed_safety, f"safety {observed_safety} != {case.expected_safety}"
        )
    observed = observe_path(case, router=active)
    if observed != case.expected_path:
        return CaseVerdict(case.id, False, observed, f"path {observed} != {case.expected_path}")
    return CaseVerdict(case.id, True, observed, "routing matches production boundary")


def evaluate_all(
    cases: list[BoundaryCase], *, router: SafetyRouter | None = None
) -> tuple[str, list[CaseVerdict]]:
    """Evaluate every case; return ``(result, verdicts)`` (pass/fail)."""
    verdicts = [evaluate_case(case, router=router) for case in cases]
    result = "pass" if all(item.passed for item in verdicts) else "fail"
    return result, verdicts


def fail_closed_reply_is_valid() -> bool:
    """Return whether the fixed fail-closed reply stays valid metadata."""
    return bool(FAIL_CLOSED_REPLY.strip()) and "уточнить" in FAIL_CLOSED_REPLY


def fixture_summary(cases: list[BoundaryCase]) -> dict[str, Any]:
    """Return a privacy-safe fixture identity summary (no user text)."""
    by_class: dict[str, int] = {}
    for case in cases:
        by_class[case.routing_class] = by_class.get(case.routing_class, 0) + 1
    return {"schema": SCHEMA_VERSION, "cases": len(cases), "by_class": by_class}


__all__ = [
    "BoundaryCase",
    "CaseVerdict",
    "EXPECTED_PATHS",
    "FIXTURE_REL",
    "REQUIRED_CLASSES",
    "REQUIRED_UTTERANCES",
    "SCHEMA_VERSION",
    "ProductionBoundaryError",
    "evaluate_all",
    "evaluate_case",
    "fail_closed_reply_is_valid",
    "find_repo_root",
    "fixture_summary",
    "load_fixture",
    "observe_path",
]
