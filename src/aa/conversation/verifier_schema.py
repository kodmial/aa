"""Claim-level structured verifier schema (Pydantic-native).

The verifier gate is model-structured semantics, never handcrafted
lexical heuristics: no keyword overlap, token stems, hand-written phrase
lists, or source-ID presence checks decide semantic support. OpenCode
native ``format=json_schema`` plus bounded native retry delivers the
object; AA code performs exactly one Pydantic validation and the
deterministic completeness checks below.

Transport contract (Gate C repair kodmial/aa#190): per-unit only. The
model emits one small boolean decision per response unit and never emits
a free-form scope string, a unit id, or an aggregate flag. AA code binds
``unit_id`` from the input unit, derives the internal scope
deterministically (``book`` when ``requires_book_evidence`` is true,
otherwise a non-book glue-compatible scope), and computes
``all_required_supported`` as the conjunction of per-unit support.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ScopeName = Literal["book", "product_meta", "conversation_glue"]

# Internal non-book scope derived when the model reports
# ``requires_book_evidence=false``. Both pure conversation glue and
# truthful assistant identity/capability statements map here; neither
# needs book evidence and both are handled identically downstream.
NON_BOOK_SCOPE: ScopeName = "conversation_glue"

# Bounded native retry for one verifier call (Gate C live repair, run
# 37538518277: p50 16.4s/p95 34.4s/max 42.7s over the 30s hard budget with
# the verifier never served and all ordinary turns collapsed to generic
# clarification). OpenCode owns this retry. One server retry halves the
# worst-case verifier latency while the AA-side strict Pydantic +
# completeness gate stays unchanged.
VERIFIER_MAX_ATTEMPTS = 1


class UnitDecision(BaseModel):
    """Provider-native per-unit verifier decision (transport only)."""

    requires_book_evidence: bool
    supported: bool
    evidence_passage_ids: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class UnitVerdict(BaseModel):
    """One verifier verdict for one ordered response unit (internal)."""

    unit_id: str = Field(min_length=1)
    scope: ScopeName
    supported: bool
    evidence_passage_ids: list[str] = Field(default_factory=list)


class GroundingResult(BaseModel):
    """Claim-level grounding outcome for one draft (internal)."""

    units: list[UnitVerdict] = Field(min_length=1)
    all_required_supported: bool
    # Deterministic AA transport metadata; never emitted by the model.
    unavailable_unit_ids: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class VerifierValidationError(ValueError):
    """Structured verifier output failed completeness validation."""


def verifier_single_json_schema() -> dict[str, object]:
    """Build the minimal per-unit transport schema for providers.

    The native hint contains only booleans plus the citation list: no
    free-form scope string, no ``unit_id`` copy, and no model-computed
    aggregate. The schema stays ``$ref``-free and omits length
    constraints (enforced in AA code); ``required`` stays as essential
    guidance. Turn-independent, never an exact-question special case.
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
        },
        "required": ["requires_book_evidence", "supported", "evidence_passage_ids"],
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
    return result


__all__ = [
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
