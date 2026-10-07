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

# Bounded native retry for one verifier call (Gate C live repair, run
# 37538518277: p50 16.4s/p95 34.4s/max 42.7s over the 30s hard budget with
# the verifier never served and all ordinary turns collapsed to generic
# clarification). OpenCode owns this retry; AA code adds at most one more
# id-completeness retry in turn_pipeline. Two server retries plus the
# AA-side retry burned up to six slow fallback model calls per flaky turn.
# One server retry halves the worst-case verifier latency while the
# AA-side strict Pydantic + completeness gate stays unchanged.
VERIFIER_MAX_ATTEMPTS = 1


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

    The transport hint is additionally minimal on purpose (Gate C run
    37538518277): ``minLength``/``minItems`` are omitted because weak
    fallback providers reject or flake on length constraints while the
    same constraints stay enforced in AA code (Pydantic ``min_length`` /
    ``min_length`` plus the exactly-one-verdict-per-unit completeness
    gate). ``enum`` and ``required`` stay because they are the essential
    guidance a weak model needs to emit a schema-valid verdict on the
    first try instead of burning retry cycles.
    """
    return {
        "type": "object",
        "properties": {
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "unit_id": {"type": "string"},
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


def verifier_single_json_schema() -> dict[str, object]:
    """Build the minimal single-unit transport schema for weak providers.

    Turn-independent hardening: a single verdict object (no array, no
    ``unit_id`` copy) is strictly easier for a weak fallback model than
    the batch array with exact ``u1``..``uN`` id copying. The native hint
    stays ``$ref``-free and omits length constraints (enforced in AA
    code); ``enum`` + ``required`` stay as essential guidance.
    """
    return {
        "type": "object",
        "properties": {
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
        "required": ["scope", "supported"],
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
    "verifier_single_json_schema",
]
