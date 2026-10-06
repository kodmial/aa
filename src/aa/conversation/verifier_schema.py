"""Claim-level structured verifier schema (Pydantic-native).

The verifier gate is model-structured semantics, never handcrafted
lexical heuristics: no keyword overlap, token stems, hand-written phrase
lists, or source-ID presence checks decide semantic support. OpenCode
native ``format=json_schema`` plus bounded native retry delivers the
object; AA code performs exactly one Pydantic validation and the
deterministic completeness checks below.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ScopeName = Literal["book", "product_meta", "conversation_glue"]

VERIFIER_MAX_ATTEMPTS = 2


class UnitVerdict(BaseModel):
    """One verifier verdict for one ordered response unit."""

    unit_id: str = Field(min_length=1)
    scope: ScopeName
    supported: bool
    evidence_passage_ids: list[str] = Field(default_factory=list)


class GroundingResult(BaseModel):
    """Claim-level grounding outcome for one draft."""

    units: list[UnitVerdict] = Field(min_length=1)
    all_required_supported: bool

    model_config = {"extra": "forbid"}


class VerifierValidationError(ValueError):
    """Structured verifier output failed completeness validation."""


def verifier_json_schema() -> dict[str, object]:
    """Build the native OpenCode JSON Schema for the verifier.

    The schema is flattened and ``$ref``-free on purpose (Gate C live
    repair): the raw Pydantic ``model_json_schema()`` emits ``$defs`` plus
    ``$ref`` for the nested ``UnitVerdict`` model, and weak fallback
    providers reject or flake on ``$ref`` (verifier never served, all
    ordinary turns collapsing to generic clarification with slow internal
    retries). The planner already uses a flat ``$ref``-free native schema
    for the same reason. AA-side validation stays strict Pydantic
    (``validate_grounding_result`` with ``extra=forbid`` and ID
    completeness); this native schema is only the OpenCode transport hint.
    """
    return {
        "type": "object",
        "properties": {
            "units": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "unit_id": {"type": "string", "minLength": 1},
                        "scope": {
                            "type": "string",
                            "enum": ["book", "product_meta", "conversation_glue"],
                        },
                        "supported": {"type": "boolean"},
                        "evidence_passage_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["unit_id", "scope", "supported"],
                },
            },
            "all_required_supported": {"type": "boolean"},
        },
        "required": ["units", "all_required_supported"],
    }


def validate_grounding_result(data: object, *, expected_unit_ids: list[str]) -> GroundingResult:
    """Pydantic-validate one native verifier object plus ID completeness.

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
    for unit_id in seen:
        if unit_id not in set(expected_unit_ids):
            raise VerifierValidationError(f"verifier returned unknown unit {unit_id!r}")
    return result


__all__ = [
    "GroundingResult",
    "ScopeName",
    "UnitVerdict",
    "VERIFIER_MAX_ATTEMPTS",
    "VerifierValidationError",
    "validate_grounding_result",
    "verifier_json_schema",
]
