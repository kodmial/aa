"""Typed answer generation/verification failure causality (kodmial/aa#436).

The historical Gate C run could not recover *why* a turn collapsed:
``generate_dialogue_reply`` swallowed every exception into ``None``,
``_verify_draft`` collapsed split errors, local guards, verifier
validation/outage and negative verdicts into one ``result=None``
shape, and the terminal ``clarification-unavailable`` /
``conversational-generation-failed`` categories were the only
diagnostics. This module carries a privacy-safe, typed causal record
for every failed generation/verification step and preserves the
*earliest* specific cause through later fallback, repair and terminal
clarification paths.

Closed-vocabulary reasons only. No prompt, reply, passage, session or
credential text ever enters a cause record: every field is a fixed
string from this module's vocabularies, a bounded constructor-derived
error-kind code, or an opaque correlation id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_REASON_LEN = 64
MAX_DETAIL_LEN = 64
MAX_ID_LEN = 64
MAX_CHAIN_LEN = 6

TELEMETRY_CAUSE_KEY = "answer_failure_cause"
TELEMETRY_CHAIN_KEY = "answer_failure_cause_chain"

# Production outcomes that represent a genuinely certified substantive
# reply reaching the user (delivery certificate + adequacy/verifier
# contract). Any other served answer_outcome (clarification, retry,
# budget/adequacy failures with natural text, ...) is NOT a product
# success even when transport delivery succeeded (#436).
PRODUCT_SUCCESS_ANSWER_OUTCOMES = frozenset(
    {
        "served",
        "narrowed-supported",
        "narrowed-compacted",
        "narrowed-adequacy",
        "narrowed-adequacy-regen",
        "conversational-generated",
    }
)

# Closed-vocabulary stages (attribution boundary).
STAGE_GENERATION = "generation"
STAGE_SPLIT = "split"
STAGE_VERIFICATION = "verification"
STAGE_EVIDENCE_INTEGRITY = "evidence-integrity"
STAGE_FINALIZATION = "finalization"

STAGES: tuple[str, ...] = (
    STAGE_GENERATION,
    STAGE_SPLIT,
    STAGE_VERIFICATION,
    STAGE_EVIDENCE_INTEGRITY,
    STAGE_FINALIZATION,
)

# Closed-vocabulary sources: a provider failure never masquerades as a
# local one and vice versa.
SOURCE_PROVIDER = "provider"
SOURCE_LOCAL = "local"
SOURCE_VERIFICATION = "verification"
SOURCE_BUDGET = "budget"
SOURCE_UNKNOWN = "unknown"

SOURCES: tuple[str, ...] = (
    SOURCE_PROVIDER,
    SOURCE_LOCAL,
    SOURCE_VERIFICATION,
    SOURCE_BUDGET,
    SOURCE_UNKNOWN,
)

# Closed-vocabulary failure reasons.
GENERATION_PROVIDER_RATE_LIMIT = "provider-rate-limit"
GENERATION_PROVIDER_ACCESS = "provider-access"
GENERATION_PROVIDER_TIMEOUT = "provider-timeout"
GENERATION_PROVIDER_TRANSIENT = "provider-transient"
GENERATION_PROVIDER_INVALID = "provider-invalid-response"
GENERATION_EXCEPTION = "generation-exception"
GENERATION_EMPTY_CANDIDATE = "generation-empty-candidate"
GUARD_LANGUAGE = "guard-language"
GUARD_INTERNAL_TERMS = "guard-internal-terms"
GUARD_ENVELOPE = "guard-envelope"
GUARD_QUOTE_BUDGET = "guard-quote-budget"
GUARD_OUTBOUND_SAFETY = "guard-outbound-safety"
SPLIT_INVALID = "split-invalid"
EVIDENCE_INTEGRITY_INVALID = "evidence-integrity-invalid"
VERIFIER_SCHEMA_INVALID = "verifier-invalid-schema"
VERIFIER_INCOMPLETE = "verifier-incomplete-verdict"
VERIFIER_UNAVAILABLE = "verifier-unavailable"
VERIFIER_SKIPPED_BUDGET = "verifier-skipped-turn-budget"
VERDICT_MISSING = "verdict-missing"
VERDICT_UNSUPPORTED = "verdict-unsupported"
VERDICT_IRRELEVANT = "verdict-irrelevant"
BUDGET_EXHAUSTED = "budget-exhausted"
REPAIR_EXHAUSTED = "repair-exhausted"
UNKNOWN = "unknown"

KNOWN_REASONS: frozenset[str] = frozenset(
    {
        GENERATION_PROVIDER_RATE_LIMIT,
        GENERATION_PROVIDER_ACCESS,
        GENERATION_PROVIDER_TIMEOUT,
        GENERATION_PROVIDER_TRANSIENT,
        GENERATION_PROVIDER_INVALID,
        GENERATION_EXCEPTION,
        GENERATION_EMPTY_CANDIDATE,
        GUARD_LANGUAGE,
        GUARD_INTERNAL_TERMS,
        GUARD_ENVELOPE,
        GUARD_QUOTE_BUDGET,
        GUARD_OUTBOUND_SAFETY,
        SPLIT_INVALID,
        EVIDENCE_INTEGRITY_INVALID,
        VERIFIER_SCHEMA_INVALID,
        VERIFIER_INCOMPLETE,
        VERIFIER_UNAVAILABLE,
        VERIFIER_SKIPPED_BUDGET,
        VERDICT_MISSING,
        VERDICT_UNSUPPORTED,
        VERDICT_IRRELEVANT,
        BUDGET_EXHAUSTED,
        REPAIR_EXHAUSTED,
        UNKNOWN,
    }
)


def _bounded(value: object, limit: int) -> str:
    return str(value or "")[:limit]


@dataclass(frozen=True)
class AnswerFailureCause:
    """One typed, privacy-safe answer failure cause."""

    reason: str
    stage: str
    source: str
    detail: str = ""
    causality_id: str = ""

    @staticmethod
    def bounded(
        reason: str, stage: str, source: str, *, detail: str = "", causality_id: str = ""
    ) -> AnswerFailureCause:
        """Build a bounded cause, normalizing out-of-vocabulary fields.

        Unknown reason/stage/source values stay explicit instead of
        being coerced into a wrong closed term: an unclassified failure
        records ``unknown`` (with its observed stage) so missing
        information is never invented into a specific label.
        """
        from aa.conversation.turn_causality import current_causality_id

        reason_v = _bounded(reason, MAX_REASON_LEN)
        stage_v = _bounded(stage, MAX_REASON_LEN)
        source_v = _bounded(source, MAX_REASON_LEN)
        if reason_v not in KNOWN_REASONS:
            reason_v = UNKNOWN
        if stage_v not in STAGES:
            stage_v = STAGE_FINALIZATION
        if source_v not in SOURCES:
            source_v = SOURCE_UNKNOWN
        correlation = _bounded(causality_id, MAX_ID_LEN)
        if not correlation:
            try:
                correlation = current_causality_id()
            except Exception:
                correlation = ""
        return AnswerFailureCause(
            reason=reason_v,
            stage=stage_v,
            source=source_v,
            detail=_bounded(detail, MAX_DETAIL_LEN),
            causality_id=correlation,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "reason": self.reason,
            "stage": self.stage,
            "source": self.source,
            "detail": self.detail,
            "causality_id": self.causality_id,
        }

    @staticmethod
    def from_payload(payload: Any) -> AnswerFailureCause | None:
        """Rebuild a bounded cause from persisted telemetry (fail-closed)."""
        if not isinstance(payload, dict):
            return None
        return AnswerFailureCause.bounded(
            str(payload.get("reason", "") or ""),
            str(payload.get("stage", "") or ""),
            str(payload.get("source", "") or ""),
            detail=str(payload.get("detail", "") or ""),
            causality_id=str(payload.get("causality_id", "") or ""),
        )


def _provider_cause_from_exception(exc: BaseException) -> AnswerFailureCause:
    """Map a provider/transport exception to a typed cause without guessing.

    Rate limits, access rejections, timeouts, transient outages and
    deterministic invalid responses keep distinct typed reasons; the
    original retirement/cancellation contract is never consumed here
    (:class:`asyncio.CancelledError` propagates through callers before
    reaching this classifier).
    """
    from aa.opencode.errors import (
        OpenCodeDeterministicError,
        OpenCodeError,
        OpenCodeProviderAccessError,
        OpenCodeRateLimitError,
        OpenCodeTimeoutError,
        OpenCodeTransientError,
    )

    detail = _bounded(type(exc).__name__, MAX_DETAIL_LEN)
    if isinstance(exc, OpenCodeRateLimitError):
        return AnswerFailureCause.bounded(
            GENERATION_PROVIDER_RATE_LIMIT, STAGE_GENERATION, SOURCE_PROVIDER, detail=detail
        )
    if isinstance(exc, OpenCodeProviderAccessError):
        return AnswerFailureCause.bounded(
            GENERATION_PROVIDER_ACCESS, STAGE_GENERATION, SOURCE_PROVIDER, detail=detail
        )
    if isinstance(exc, OpenCodeTimeoutError) or isinstance(exc, TimeoutError):
        return AnswerFailureCause.bounded(
            GENERATION_PROVIDER_TIMEOUT, STAGE_GENERATION, SOURCE_PROVIDER, detail=detail
        )
    if isinstance(exc, OpenCodeDeterministicError):
        return AnswerFailureCause.bounded(
            GENERATION_PROVIDER_INVALID, STAGE_GENERATION, SOURCE_PROVIDER, detail=detail
        )
    if isinstance(exc, OpenCodeTransientError):
        return AnswerFailureCause.bounded(
            GENERATION_PROVIDER_TRANSIENT, STAGE_GENERATION, SOURCE_PROVIDER, detail=detail
        )
    if isinstance(exc, OpenCodeError):
        return AnswerFailureCause.bounded(
            GENERATION_EXCEPTION, STAGE_GENERATION, SOURCE_PROVIDER, detail=detail
        )
    return AnswerFailureCause.bounded(
        GENERATION_EXCEPTION, STAGE_GENERATION, SOURCE_PROVIDER, detail=detail
    )


def cause_from_exception(
    exc: BaseException, *, stage: str = STAGE_GENERATION
) -> AnswerFailureCause:
    """Classify one caught exception into a typed answer-failure cause.

    Verifier validation errors (schema/completeness/citation/checksum
    gates, all enforced locally) stay distinct from transport/verifier
    unavailability; unknown exception classes record ``unknown`` with
    their bounded type code in ``detail`` instead of a guessed label.
    """
    from aa.conversation.verifier_schema import VerifierValidationError

    detail = _bounded(type(exc).__name__, MAX_DETAIL_LEN)
    if isinstance(exc, VerifierValidationError):
        return AnswerFailureCause.bounded(
            VERIFIER_SCHEMA_INVALID, stage, SOURCE_LOCAL, detail=detail
        )
    cause = _provider_cause_from_exception(exc)
    if stage == STAGE_VERIFICATION and cause.reason in (
        GENERATION_EXCEPTION,
        GENERATION_PROVIDER_TRANSIENT,
        GENERATION_PROVIDER_TIMEOUT,
    ):
        return AnswerFailureCause.bounded(
            VERIFIER_UNAVAILABLE, stage, SOURCE_VERIFICATION, detail=detail
        )
    return AnswerFailureCause.bounded(
        cause.reason, stage, cause.source, detail=cause.detail, causality_id=cause.causality_id
    )


def attach_answer_failure_cause(
    telemetry: dict[str, Any],
    cause: AnswerFailureCause,
) -> None:
    """Record one typed cause into safe turn telemetry.

    The *first* cause recorded stays in ``answer_failure_cause`` and is
    never overwritten by later fallback/repair/terminal stages;
    subsequent distinct causes append to ``answer_failure_cause_chain``
    (bounded). This preserves the earliest specific cause all the way to
    the terminal ``clarification-unavailable`` telemetry.
    """
    if not isinstance(telemetry, dict):
        return
    record = cause.to_dict()
    existing = telemetry.get(TELEMETRY_CAUSE_KEY)
    if not (isinstance(existing, dict) and str(existing.get("reason", "") or "").strip()):
        telemetry[TELEMETRY_CAUSE_KEY] = record
        telemetry[TELEMETRY_CHAIN_KEY] = [record]
        return
    chain = telemetry.get(TELEMETRY_CHAIN_KEY)
    if not isinstance(chain, list):
        chain = []
    existing_record = AnswerFailureCause.from_payload(existing)
    same_as_latest = bool(chain) and isinstance(chain[-1], dict) and chain[-1] == record
    same_as_initial = existing_record is not None and existing_record.to_dict() == record
    if not same_as_latest and not same_as_initial:
        chain.append(record)
    if len(chain) > MAX_CHAIN_LEN:
        del chain[: len(chain) - MAX_CHAIN_LEN]
    telemetry[TELEMETRY_CHAIN_KEY] = chain


def cause_for_empty_candidate(
    *, stage: str = STAGE_GENERATION, causality_id: str = ""
) -> AnswerFailureCause:
    return AnswerFailureCause.bounded(
        GENERATION_EMPTY_CANDIDATE, stage, SOURCE_LOCAL, causality_id=causality_id
    )


def failure_reason_from_telemetry(snapshot: Any) -> str:
    """Return the bounded typed failure reason for one telemetry snapshot.

    Prefers the typed ``answer_failure_cause.reason`` recorded by this
    module; falls back to the bounded legacy ``failure_category``. An
    empty string means genuinely unknown, never a fabricated cause.
    """
    if not isinstance(snapshot, dict):
        return ""
    cause = snapshot.get(TELEMETRY_CAUSE_KEY)
    if isinstance(cause, dict):
        reason = _bounded(cause.get("reason"), MAX_REASON_LEN)
        if reason:
            return reason
    return _bounded(snapshot.get("failure_category"), MAX_REASON_LEN)


def verifier_locale_reason_for_message(message: str) -> str | None:
    """Classify a pre-HTTP verifier rejection message into a typed reason.

    ``run_verifier``/``run_batched_verifier`` raise
    :class:``VerifierValidationError`` deterministically BEFORE any
    HTTP call for three locally-provable cases: no response units, a
    non-positive verifier turn budget, and evidence-pack integrity.
    Collapsing these into a generic verifier-invalid/unavailable label
    hid the real boundary (the historical ms-latency ``unavailable``
    verifier outcomes); they keep distinct typed reasons here.
    """
    normalized = (message or "").strip()
    if "at least one response unit" in normalized:
        return GENERATION_EMPTY_CANDIDATE
    if "turn budget must be > 0" in normalized:
        return VERIFIER_SKIPPED_BUDGET
    if "evidence integrity failure" in normalized:
        return EVIDENCE_INTEGRITY_INVALID
    return None


def guard_cause_for_verifier_locale_message(message: str) -> AnswerFailureCause:
    """Build the typed local cause for a pre-HTTP verifier rejection.

    Unrecognized validation messages stay the explicit
    ``verifier-invalid-schema`` (model/parse validation), never a
    guessed provider or outage label.
    """
    reason = verifier_locale_reason_for_message(message) or VERIFIER_SCHEMA_INVALID
    return AnswerFailureCause.bounded(
        reason, STAGE_VERIFICATION, SOURCE_LOCAL, detail="verifier-local"
    )


def terminal_unknown_cause(existing: Any, *, stage: str = STAGE_FINALIZATION) -> AnswerFailureCause:
    """Return the authoritative existing cause, else an explicit unknown.

    Terminal stages (for example ``clarification-unavailable``) reuse the
    earliest recorded typed cause; when none was recorded they keep the
    honest ``unknown`` label with the boundary stage instead of a guessed
    reason.
    """
    if isinstance(existing, dict) and _bounded(existing.get("reason"), MAX_REASON_LEN):
        existing_cause = AnswerFailureCause.from_payload(existing)
        if existing_cause is not None:
            return existing_cause
    return AnswerFailureCause.bounded(UNKNOWN, stage, SOURCE_UNKNOWN, detail="no-specific-cause")


__all__ = [
    "AnswerFailureCause",
    "BUDGET_EXHAUSTED",
    "EVIDENCE_INTEGRITY_INVALID",
    "GENERATION_EMPTY_CANDIDATE",
    "GENERATION_EXCEPTION",
    "GENERATION_PROVIDER_ACCESS",
    "GENERATION_PROVIDER_INVALID",
    "GENERATION_PROVIDER_RATE_LIMIT",
    "GENERATION_PROVIDER_TIMEOUT",
    "GENERATION_PROVIDER_TRANSIENT",
    "GUARD_ENVELOPE",
    "GUARD_INTERNAL_TERMS",
    "GUARD_LANGUAGE",
    "GUARD_OUTBOUND_SAFETY",
    "GUARD_QUOTE_BUDGET",
    "KNOWN_REASONS",
    "MAX_CHAIN_LEN",
    "PRODUCT_SUCCESS_ANSWER_OUTCOMES",
    "REPAIR_EXHAUSTED",
    "SOURCES",
    "SPLIT_INVALID",
    "STAGES",
    "STAGE_EVIDENCE_INTEGRITY",
    "STAGE_FINALIZATION",
    "STAGE_GENERATION",
    "STAGE_SPLIT",
    "STAGE_VERIFICATION",
    "TELEMETRY_CAUSE_KEY",
    "TELEMETRY_CHAIN_KEY",
    "UNKNOWN",
    "VERDICT_IRRELEVANT",
    "VERDICT_MISSING",
    "VERDICT_UNSUPPORTED",
    "VERIFIER_INCOMPLETE",
    "VERIFIER_SCHEMA_INVALID",
    "VERIFIER_SKIPPED_BUDGET",
    "VERIFIER_UNAVAILABLE",
    "attach_answer_failure_cause",
    "cause_for_empty_candidate",
    "cause_from_exception",
    "failure_reason_from_telemetry",
    "guard_cause_for_verifier_locale_message",
    "terminal_unknown_cause",
    "verifier_locale_reason_for_message",
]
