"""AA conversation package (issue #9).

Deterministic Russian-first grounded conversational runtime. The
orchestrator state machine lives in :mod:`aa.conversation.orchestrator`.
"""

from __future__ import annotations

from aa.conversation.orchestrator import (
    COVERAGE_SCHEMA_VERSION,
    FAIL_CLOSED_REPLY,
    RUNTIME_VERSION,
    SUPPORT_SCHEMA_VERSION,
    AnswerUnit,
    CoverageResult,
    EvidencePack,
    GroundedResponse,
    ResponseUnit,
    TurnDiagnostics,
    TurnFailed,
    build_local_plan_payload,
    is_substantive,
    validate_coverage_payload,
    validate_support_payload,
)

__all__ = [
    "FAIL_CLOSED_REPLY",
    "COVERAGE_SCHEMA_VERSION",
    "RUNTIME_VERSION",
    "SUPPORT_SCHEMA_VERSION",
    "AnswerUnit",
    "CoverageResult",
    "EvidencePack",
    "GroundedResponse",
    "ResponseUnit",
    "TurnDiagnostics",
    "TurnFailed",
    "build_local_plan_payload",
    "is_substantive",
    "validate_coverage_payload",
    "validate_support_payload",
]
