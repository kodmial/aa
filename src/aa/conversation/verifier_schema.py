"""Claim-level structured verifier schema (Pydantic-native, issue #268).

The verifier gate is model-structured semantics, never handcrafted
lexical heuristics: no keyword overlap, token stems, hand-written phrase
lists, or source-ID presence checks decide semantic support.

Transport contract: per-unit only. Each invocation returns both
claim-support and intent-relevance verdicts for exactly one response
unit, so groundedness and answer relevance are judged by the same model
call against the same Evidence Pack. AA code binds ``unit_id`` from the
input unit, derives the internal scope deterministically, and computes
the turn aggregates as conjunctions. The model never copies ids and never
computes an aggregate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictBool

ScopeName = Literal["book", "product_meta", "conversation_glue"]

ClaimOrigin = Literal[
    "book_claim",
    "user_report",
    "assistant_capability",
    "conversation_glue",
    "safety_override",
]
"""Typed claim/quote provenance contract (kodmial/aa#308).

- ``book_claim``: substantive AA/book claim, requires actual canonical
  book support and citations;
- ``user_report``: attributed statement or verbatim quote demonstrably
  found in a concrete HumanMessage; never introduces inference about
  causes, motives, diagnosis, efficacy, or advice;
- ``assistant_capability``: truthful system/product capability grounded
  only in authoritative product instructions;
- ``conversation_glue``: no independent substantive assertion;
- ``safety_override``: explicit allowed safety-policy outcome, not a
  book-grounded claim.
"""

# Internal non-book scope derived when the model reports
# ``requires_book_evidence=false``. Both pure conversation glue and
# truthful assistant identity/capability statements map here; neither
# needs book evidence and both are handled identically downstream.
NON_BOOK_SCOPE: ScopeName = "conversation_glue"

# Bounded native retry for one verifier call. OpenCode owns this retry.
VERIFIER_MAX_ATTEMPTS = 1


class UnitDecision(BaseModel):
    """Provider-native per-unit verifier decision (transport only).

    Every field requires an explicit strict boolean: a missing, null, or
    wrong-type ``addresses_intent`` fails closed even when the unit is
    book-supported with valid citations. Support never implies relevance.

    ``claim_origin`` carries the model-led semantic origin classification
    (kodmial/aa#308). It is optional on the wire so older providers stay
    parseable; when absent AA code derives the origin deterministically
    from ``requires_book_evidence`` (true -> ``book_claim``, false ->
    ``conversation_glue``) and the deterministic provenance gate binds
    it to offsets and roles. ``origin_ref`` is extensible provenance for
    the origin (user message slice, book anchor, capability source, or
    policy marker); a quote mark alone never sets the origin.
    """

    requires_book_evidence: bool
    supported: bool
    evidence_passage_ids: list[str] = Field(default_factory=list)
    addresses_intent: StrictBool
    claim_origin: ClaimOrigin | None = None
    origin_ref: dict[str, object] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class UnitVerdict(BaseModel):
    """One verifier verdict for one ordered response unit (internal)."""

    unit_id: str = Field(min_length=1)
    scope: ScopeName
    supported: bool
    evidence_passage_ids: list[str] = Field(default_factory=list)
    addresses_intent: bool = Field(default=False)
    origin: ClaimOrigin = "conversation_glue"
    origin_ref: dict[str, object] = Field(default_factory=dict)


class GroundingResult(BaseModel):
    """Claim-level grounding outcome for one draft (internal)."""

    verified: bool = Field(default=True)
    units: list[UnitVerdict] = Field(min_length=1)
    all_required_supported: bool
    # Deterministic AA transport metadata; never emitted by the model.
    unavailable_unit_ids: list[str] = Field(default_factory=list)
    # Turn-level relevance aggregate derived deterministically from the
    # per-unit model verdicts in the same invocation round: True when at
    # least one supported book unit addresses the resolved intent, or when
    # the turn carries no book requirement at all.
    answer_relevant: bool = Field(default=False)
    relevance_category: str = Field(default="")

    model_config = {"extra": "forbid"}


class VerifierValidationError(ValueError):
    """Structured verifier output failed completeness validation."""


def verifier_single_json_schema() -> dict[str, object]:
    """Build the minimal per-unit transport schema for providers.

    The native hint contains booleans plus the citation list: no
    free-form scope string, no ``unit_id`` copy, and no model-computed
    aggregate. ``addresses_intent`` records whether the unit addresses
    the resolved user intent, judged against the same Evidence Pack in
    the same call. Length constraints are enforced in AA code.
    """
    return {
        "type": "object",
        "properties": {
            "requires_book_evidence": {"type": "boolean"},
            "supported": {"type": "boolean"},
            "evidence_passage_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "addresses_intent": {"type": "boolean"},
            # kodmial/aa#308: optional model-led claim-origin hint plus
            # extensible origin_ref provenance. Kept intentionally
            # enum-free: the provider-facing hint stays minimal/robust
            # across weak fallback paths, while AA code strictly
            # validates the closed origin vocabulary via Pydantic
            # (fail closed on unknown origins).
            "claim_origin": {"type": "string"},
            "origin_ref": {"type": "object"},
        },
        "required": [
            "requires_book_evidence",
            "supported",
            "evidence_passage_ids",
            "addresses_intent",
        ],
    }


def validate_unit_decision(data: object) -> UnitDecision:
    """Pydantic-validate one native per-unit verifier decision."""
    from pydantic import ValidationError as PydanticValidationError

    if isinstance(data, UnitDecision):
        return data
    if isinstance(data, dict):
        try:
            return UnitDecision.model_validate(data)
        except PydanticValidationError as exc:
            raise VerifierValidationError(f"verifier output invalid: {exc}") from exc
    raise VerifierValidationError("verifier output is not a structured object")


def _derive_turn_relevance(units: list[UnitVerdict]) -> tuple[bool, str]:
    """Derive turn relevance deterministically from per-unit model verdicts.

    No lexical heuristics: the model already judged each unit. The turn
    is relevant when every supported book unit addresses the resolved
    intent, or when no unit requires book evidence at all (pure
    conversational turn). Requiring every supported book unit (not just
    one) rejects padding, repetition and off-topic digressions served
    alongside one relevant sentence (kodmial/aa#286): a single relevant
    sentence no longer passes the entire response. Pure empathy/glue
    units carry no substantive claim and are not required to address
    the intent individually.
    """
    if not units:
        return False, "no-units"
    needs_book = any(verdict.scope == "book" for verdict in units)
    if not needs_book:
        return True, "glue-no-book-required"
    supported_book = [verdict for verdict in units if verdict.scope == "book" and verdict.supported]
    if not supported_book:
        return False, "irrelevant-citation"
    for verdict in supported_book:
        if not verdict.addresses_intent:
            return False, "irrelevant-citation"
    return True, ""


def validate_grounding_result(data: object, *, expected_unit_ids: list[str]) -> GroundingResult:
    """Pydantic-validate one internal grounding result plus ID completeness.

    Every supplied ``unit_id`` must receive exactly one verdict: missing,
    duplicate, or unknown IDs invalidate the verification.
    """
    from pydantic import ValidationError as PydanticValidationError

    if isinstance(data, GroundingResult):
        result = data
    elif isinstance(data, dict):
        try:
            result = GroundingResult.model_validate(data)
        except PydanticValidationError as exc:
            raise VerifierValidationError(f"verifier output invalid: {exc}") from exc
    else:
        raise VerifierValidationError("verifier output is not a structured object")
    if not expected_unit_ids:
        raise VerifierValidationError("verifier needs at least one response unit")
    seen: dict[str, int] = {}
    for verdict in result.units:
        seen[verdict.unit_id] = seen.get(verdict.unit_id, 0) + 1
    if len(result.units) != len(expected_unit_ids):
        raise VerifierValidationError(
            f"verifier must return exactly one verdict per unit: "
            f"expected {len(expected_unit_ids)}, got {len(result.units)}"
        )
    for unit_id in expected_unit_ids:
        if seen.get(unit_id, 0) != 1:
            raise VerifierValidationError(
                f"verifier verdict for {unit_id!r} is missing or duplicated"
            )
    expected = set(expected_unit_ids)
    for unit_id in seen:
        if unit_id not in expected:
            raise VerifierValidationError(f"verifier returned unknown unit {unit_id!r}")
    for unit_id in result.unavailable_unit_ids:
        if unit_id not in expected:
            raise VerifierValidationError(
                f"verifier unavailable metadata contains unknown unit {unit_id!r}"
            )
        match = next((item for item in result.units if item.unit_id == unit_id), None)
        if match is None or match.supported:
            raise VerifierValidationError(f"unavailable verifier unit {unit_id!r} must fail closed")
    relevant, category = _derive_turn_relevance(list(result.units))
    return GroundingResult(
        verified=bool(result.verified),
        units=list(result.units),
        all_required_supported=bool(result.all_required_supported),
        unavailable_unit_ids=list(result.unavailable_unit_ids),
        answer_relevant=relevant,
        relevance_category=category,
    )


__all__ = [
    "ClaimOrigin",
    "GroundingResult",
    "NON_BOOK_SCOPE",
    "ScopeName",
    "UnitDecision",
    "UnitVerdict",
    "VERIFIER_MAX_ATTEMPTS",
    "VerifierValidationError",
    "validate_grounding_result",
    "validate_unit_decision",
    "verifier_single_json_schema",
]
