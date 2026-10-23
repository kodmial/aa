"""Typed answer-failure causality taxonomy and pipeline wiring (kodmial/aa#436).

History: a delivered service error could not be attributed because
``generate_dialogue_reply`` swallowed every exception (including provider
429 and cancellation) into ``None`` and ``_verify_draft`` collapsed
split/local/pre-HTTP-verifier/transport/negative-verdict boundaries into
one ``None`` shape. These tests pin the closed vocabulary, the
first-cause-preserving chain, the exception mapping, the propagation
contract and the privacy guarantee (no text, only fixed terms).

No prompt, reply or canonical-book text is committed here; artificial
wording only.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import aa.conversation.turn_pipeline as turn_pipeline
from aa.conversation.answer_causality import (
    EVIDENCE_INTEGRITY_INVALID,
    GENERATION_EMPTY_CANDIDATE,
    GENERATION_PROVIDER_RATE_LIMIT,
    GUARD_INTERNAL_TERMS,
    GUARD_LANGUAGE,
    KNOWN_REASONS,
    SOURCES,
    SPLIT_INVALID,
    STAGE_GENERATION,
    STAGE_VERIFICATION,
    STAGES,
    UNKNOWN,
    VERIFIER_SCHEMA_INVALID,
    VERIFIER_SKIPPED_BUDGET,
    AnswerFailureCause,
    attach_answer_failure_cause,
    cause_for_empty_candidate,
    cause_from_exception,
    failure_reason_from_telemetry,
    first_answer_failure_cause,
    guard_cause_for_verifier_locale_message,
    terminal_unknown_cause,
    verifier_locale_reason_for_message,
)
from aa.conversation.turn_pipeline import generate_dialogue_reply
from aa.conversation.verifier_schema import GroundingResult, UnitVerdict, VerifierValidationError
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeRateLimitError,
    OpenCodeTransientError,
)


class _StubModel:
    pass


# ---------------------------------------------------------------------------
# Vocabulary bounds.
# ---------------------------------------------------------------------------


def test_bounded_normalizes_out_of_vocabulary_fields() -> None:
    cause = AnswerFailureCause.bounded("bogus-reason", "bogus-stage", "bogus-source")
    assert cause.reason == UNKNOWN
    assert cause.stage == "finalization"
    assert cause.source == "unknown"
    assert cause.reason in KNOWN_REASONS
    assert cause.stage in STAGES
    assert cause.source in SOURCES


def test_bounded_truncates_long_detail_and_keeps_known_terms() -> None:
    cause = AnswerFailureCause.bounded(GUARD_LANGUAGE, STAGE_GENERATION, "local", detail="x" * 200)
    assert cause.reason == GUARD_LANGUAGE
    assert len(cause.detail) <= 64
    assert len(cause.causality_id) <= 64


def test_from_payload_roundtrip_and_fail_closed() -> None:
    cause = AnswerFailureCause.bounded(SPLIT_INVALID, "split", "local", detail="bad-unit")
    rebuilt = AnswerFailureCause.from_payload(cause.to_dict())
    assert rebuilt is not None
    assert rebuilt == cause
    assert AnswerFailureCause.from_payload("junk") is None
    assert AnswerFailureCause.from_payload(None) is None


# ---------------------------------------------------------------------------
# First-cause-preserving chain.
# ---------------------------------------------------------------------------


def test_first_cause_is_preserved_later_distinct_causes_extend_chain() -> None:
    telemetry: dict[str, Any] = {}
    attach_answer_failure_cause(
        telemetry,
        AnswerFailureCause.bounded(
            GENERATION_PROVIDER_RATE_LIMIT, STAGE_GENERATION, "provider", detail="429"
        ),
    )
    attach_answer_failure_cause(
        telemetry,
        AnswerFailureCause.bounded(VERIFIER_SKIPPED_BUDGET, STAGE_VERIFICATION, "local"),
    )
    attach_answer_failure_cause(
        telemetry,
        AnswerFailureCause.bounded(VERIFIER_SKIPPED_BUDGET, STAGE_VERIFICATION, "local"),
    )
    first = telemetry["answer_failure_cause"]
    assert first["reason"] == GENERATION_PROVIDER_RATE_LIMIT
    chain = telemetry["answer_failure_cause_chain"]
    assert len(chain) == 2
    assert chain[-1]["reason"] == VERIFIER_SKIPPED_BUDGET


def test_attach_preserves_the_initial_local_guard_cause() -> None:
    """Later fallback/repair causes cannot replace a local-guard attribution."""
    telemetry: dict[str, Any] = {}
    attach_answer_failure_cause(
        telemetry,
        AnswerFailureCause.bounded(GUARD_INTERNAL_TERMS, STAGE_VERIFICATION, "local"),
    )
    attach_answer_failure_cause(
        telemetry,
        AnswerFailureCause.bounded(VERIFIER_SKIPPED_BUDGET, STAGE_VERIFICATION, "local"),
    )
    attach_answer_failure_cause(
        telemetry,
        AnswerFailureCause.bounded("verdict-unsupported", STAGE_VERIFICATION, "verification"),
    )
    assert telemetry["answer_failure_cause"]["reason"] == GUARD_INTERNAL_TERMS
    assert telemetry["answer_failure_cause_chain"][0]["reason"] == GUARD_INTERNAL_TERMS
    earlier = first_answer_failure_cause(telemetry)
    assert earlier is not None
    assert earlier["reason"] == GUARD_INTERNAL_TERMS


def test_first_answer_failure_cause_falls_back_to_the_chain() -> None:
    telemetry = {
        "answer_failure_cause_chain": [
            AnswerFailureCause.bounded(GUARD_LANGUAGE, STAGE_VERIFICATION, "local").to_dict(),
            AnswerFailureCause.bounded(SPLIT_INVALID, "split", "local").to_dict(),
        ]
    }
    cause = first_answer_failure_cause(telemetry)
    assert cause is not None
    assert cause["reason"] == GUARD_LANGUAGE
    assert first_answer_failure_cause({}) is None
    assert first_answer_failure_cause(None) is None
    assert first_answer_failure_cause({"answer_failure_cause": "junk"}) is None


def test_local_guard_reason_survives_primary_key_replacement() -> None:
    """The admission reads the immutable chain even if a caller replaces
    the primary cause without going through ``_attach``."""
    telemetry = {
        "answer_failure_cause": AnswerFailureCause.bounded(
            SPLIT_INVALID, "split", "local"
        ).to_dict(),
        "answer_failure_cause_chain": [
            AnswerFailureCause.bounded(GUARD_INTERNAL_TERMS, STAGE_VERIFICATION, "local").to_dict(),
            AnswerFailureCause.bounded(SPLIT_INVALID, "split", "local").to_dict(),
        ],
    }
    assert turn_pipeline._typed_local_guard_reason(telemetry) == GUARD_INTERNAL_TERMS


def test_chain_is_bounded_and_keeps_earliest() -> None:
    telemetry: dict[str, Any] = {}
    reasons = [
        GUARD_LANGUAGE,
        "provider-timeout",
        "provider-transient",
        SPLIT_INVALID,
        GENERATION_EMPTY_CANDIDATE,
        EVIDENCE_INTEGRITY_INVALID,
        VERIFIER_SCHEMA_INVALID,
        UNKNOWN,
    ]
    for idx, reason in enumerate(reasons):
        attach_answer_failure_cause(
            telemetry,
            AnswerFailureCause.bounded(reason, STAGE_GENERATION, "local", detail=f"d{idx}"),
        )
    chain = telemetry["answer_failure_cause_chain"]
    assert len(chain) <= 6
    assert chain[0]["reason"] == GUARD_LANGUAGE
    assert chain[-1]["reason"] == UNKNOWN
    assert telemetry["answer_failure_cause"]["reason"] == GUARD_LANGUAGE


def test_attach_ignores_non_dict_telemetry() -> None:
    holder: Any = "not-a-dict"
    attach_answer_failure_cause(
        holder, AnswerFailureCause.bounded(GUARD_LANGUAGE, STAGE_GENERATION, "local")
    )


def test_failure_reason_from_telemetry_prefers_typed_cause() -> None:
    telemetry = {
        "failure_category": "legacy-category",
        "answer_failure_cause": AnswerFailureCause.bounded(
            GUARD_LANGUAGE, STAGE_GENERATION, "local"
        ).to_dict(),
    }
    assert failure_reason_from_telemetry(telemetry) == GUARD_LANGUAGE
    assert failure_reason_from_telemetry({"failure_category": "legacy"}) == "legacy"
    assert failure_reason_from_telemetry({}) == ""


# ---------------------------------------------------------------------------
# Exception classification.
# ---------------------------------------------------------------------------


def test_rate_limit_maps_to_typed_provider_cause() -> None:
    cause = cause_from_exception(OpenCodeRateLimitError("opencode rate limited"))
    assert cause.reason == GENERATION_PROVIDER_RATE_LIMIT
    assert cause.source == "provider"
    assert cause.stage == STAGE_GENERATION


def test_transient_and_deterministic_stay_distinct() -> None:
    transient = cause_from_exception(OpenCodeTransientError("boom"))
    deterministic = cause_from_exception(OpenCodeDeterministicError("bad request"))
    assert transient.reason == "provider-transient"
    assert deterministic.reason == "provider-invalid-response"


def test_verifier_stage_transport_error_maps_to_verifier_unavailable() -> None:
    cause = cause_from_exception(OpenCodeTransientError("stream dropped"), stage="verification")
    assert cause.reason == "verifier-unavailable"
    assert cause.source == "verification"
    assert cause.stage == STAGE_VERIFICATION


def test_unknown_exception_keeps_bounded_type_detail() -> None:
    cause = cause_from_exception(RuntimeError("текст запроса и отрывок книги здесь"))
    assert cause.reason == "generation-exception"
    assert cause.detail == "RuntimeError"
    assert "текст запроса" not in str(cause)


def test_cause_for_empty_candidate() -> None:
    cause = cause_for_empty_candidate()
    assert cause.reason == GENERATION_EMPTY_CANDIDATE
    assert cause.source == "local"


def test_verifier_locale_message_classification() -> None:
    assert (
        verifier_locale_reason_for_message("verifier needs at least one response unit")
        == GENERATION_EMPTY_CANDIDATE
    )
    assert (
        verifier_locale_reason_for_message("verifier turn budget must be > 0")
        == VERIFIER_SKIPPED_BUDGET
    )
    assert (
        verifier_locale_reason_for_message("evidence integrity failure detected")
        == EVIDENCE_INTEGRITY_INVALID
    )
    assert verifier_locale_reason_for_message("schema mismatch") is None
    cause = guard_cause_for_verifier_locale_message("weird message")
    assert cause.reason == VERIFIER_SCHEMA_INVALID
    assert cause.source == "local"


def test_terminal_unknown_cause_reuses_existing() -> None:
    existing = AnswerFailureCause.bounded(
        GENERATION_PROVIDER_RATE_LIMIT, STAGE_GENERATION, "provider"
    )
    assert terminal_unknown_cause(existing.to_dict()) == existing
    fallback = terminal_unknown_cause(None)
    assert fallback.reason == UNKNOWN
    assert fallback.detail == "no-specific-cause"


# ---------------------------------------------------------------------------
# Privacy: cause records never carry free text.
# ---------------------------------------------------------------------------


def test_cause_record_is_privacy_safe_even_with_text_input() -> None:
    cause = cause_from_exception(ValueError("текст запроса и отрывок книги здесь"))
    record = cause.to_dict()
    assert set(record) <= {"reason", "stage", "source", "detail", "causality_id"}
    for value in record.values():
        assert isinstance(value, str)
        assert len(value) <= 64
    assert "текст запроса" not in record["detail"]
    assert record["reason"] in KNOWN_REASONS
    assert record["stage"] in STAGES
    assert record["source"] in SOURCES


# ---------------------------------------------------------------------------
# Pipeline wiring: generate_dialogue_reply.
# ---------------------------------------------------------------------------


def _patch_generate_draft(factory: Any) -> Any:
    import aa.conversation.answer_node as answer_node

    original = answer_node.generate_draft

    async def _replacement(*args: Any, **kwargs: Any) -> Any:
        return await factory(*args, **kwargs)

    answer_node.generate_draft = _replacement
    return original


def _restore_generate_draft(original: Any) -> None:
    import aa.conversation.answer_node as answer_node

    answer_node.generate_draft = original


async def test_rate_limit_propagates_not_swallowed() -> None:
    async def _raise_429(*args: Any, **kwargs: Any) -> str:
        raise OpenCodeRateLimitError("opencode rate limited (429)")

    original = _patch_generate_draft(_raise_429)
    try:
        telemetry: dict[str, Any] = {}
        with pytest.raises(OpenCodeRateLimitError):
            await generate_dialogue_reply(
                user_message="тяга вечером, что делать?",
                summary="",
                recent=[],
                answer_model=_StubModel(),
                telemetry=telemetry,
            )
        assert "answer_failure_cause" not in telemetry
    finally:
        _restore_generate_draft(original)


async def test_cancellation_propagates_not_swallowed() -> None:
    async def _cancel(*args: Any, **kwargs: Any) -> str:
        raise asyncio.CancelledError()

    original = _patch_generate_draft(_cancel)
    try:
        with pytest.raises(asyncio.CancelledError):
            await generate_dialogue_reply(
                user_message="тяга вечером",
                summary="",
                recent=[],
                answer_model=_StubModel(),
            )
    finally:
        _restore_generate_draft(original)


async def test_empty_candidate_records_typed_cause() -> None:
    async def _empty(*args: Any, **kwargs: Any) -> str:
        return "   "

    original = _patch_generate_draft(_empty)
    try:
        telemetry: dict[str, Any] = {}
        result = await generate_dialogue_reply(
            user_message="тяга вечером",
            summary="",
            recent=[],
            answer_model=_StubModel(),
            telemetry=telemetry,
        )
        assert result is None
        assert telemetry["answer_failure_cause"]["reason"] == GENERATION_EMPTY_CANDIDATE
        assert len(telemetry["answer_failure_cause"]) <= 5
    finally:
        _restore_generate_draft(original)


async def test_non_cyrillic_candidate_records_language_guard() -> None:
    async def _english(*args: Any, **kwargs: Any) -> str:
        return "This is artificial English filler without any Cyrillic characters."

    original = _patch_generate_draft(_english)
    try:
        telemetry: dict[str, Any] = {}
        result = await generate_dialogue_reply(
            user_message="тяга вечером",
            summary="",
            recent=[],
            answer_model=_StubModel(),
            telemetry=telemetry,
        )
        assert result is None
        assert telemetry["answer_failure_cause"]["reason"] == GUARD_LANGUAGE
        assert telemetry["answer_failure_cause"]["source"] == "local"
    finally:
        _restore_generate_draft(original)


# ---------------------------------------------------------------------------
# Pipeline wiring: _verify_draft.
# ---------------------------------------------------------------------------


async def test_verify_draft_empty_units_record_split_cause() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    telemetry: dict[str, Any] = {}
    units, result, ok = await _verify_draft(
        "   ",
        [],
        verifier_model=_StubModel(),
        telemetry=telemetry,
    )
    assert not ok and not units and result is None
    assert telemetry["answer_failure_cause"]["reason"] == GENERATION_EMPTY_CANDIDATE
    assert telemetry["answer_failure_cause"]["stage"] == "split"


async def test_verify_draft_non_cyrillic_records_language_guard() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    telemetry: dict[str, Any] = {}
    units, result, ok = await _verify_draft(
        "Plain artificial English filler sentence. Another filler sentence.",
        [{"source_id": "s1"}],
        verifier_model=_StubModel(),
        telemetry=telemetry,
    )
    assert not ok and result is None
    assert telemetry["answer_failure_cause"]["reason"] == GUARD_LANGUAGE
    assert telemetry["answer_failure_cause"]["source"] == "local"


async def test_verify_draft_verifier_budget_zero_recorded_pre_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The historical defect: a 0.0 budget fails before any HTTP call but
    was reported as generic ``verifier_outcome=unavailable``."""
    from aa.conversation.turn_pipeline import _verify_draft

    telemetry: dict[str, Any] = {}

    async def _run_verifier(*args: Any, **kwargs: Any) -> Any:
        raise VerifierValidationError("verifier turn budget must be > 0")

    monkeypatch.setattr(turn_pipeline, "run_verifier", _run_verifier)
    units, result, ok = await _verify_draft(
        "Первое предложение. Второе предложение.",
        [{"source_id": "s1"}],
        verifier_model=_StubModel(),
        turn_budget_s=0.0,
        telemetry=telemetry,
    )
    _ = units
    assert not ok and result is None
    assert telemetry["answer_failure_cause"]["reason"] == VERIFIER_SKIPPED_BUDGET
    assert telemetry["answer_failure_cause"]["source"] == "local"
    assert telemetry["answer_failure_cause"]["stage"] == "verification"


async def test_verify_draft_unclassified_validation_stays_local_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    telemetry: dict[str, Any] = {}

    async def _run_verifier(*args: Any, **kwargs: Any) -> Any:
        raise VerifierValidationError("unexpected validation detail")

    monkeypatch.setattr(turn_pipeline, "run_verifier", _run_verifier)
    units, result, ok = await _verify_draft(
        "Первое предложение. Второе предложение.",
        [{"source_id": "s1"}],
        verifier_model=_StubModel(),
        telemetry=telemetry,
    )
    assert not ok and result is None
    assert units
    assert telemetry["answer_failure_cause"]["reason"] == VERIFIER_SCHEMA_INVALID
    assert telemetry["answer_failure_cause"]["source"] == "local"
    assert telemetry["answer_failure_cause"]["stage"] == "verification"


async def test_verify_draft_transport_error_maps_to_verifier_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    telemetry: dict[str, Any] = {}

    async def _run_verifier(*args: Any, **kwargs: Any) -> Any:
        raise OpenCodeTransientError("stream dropped")

    monkeypatch.setattr(turn_pipeline, "run_verifier", _run_verifier)
    units, result, ok = await _verify_draft(
        "Первое предложение. Второе предложение.",
        [{"source_id": "s1"}],
        verifier_model=_StubModel(),
        telemetry=telemetry,
    )
    assert not ok and result is None
    assert telemetry["answer_failure_cause"]["reason"] == "verifier-unavailable"
    assert telemetry["answer_failure_cause"]["source"] == "verification"


def _grounding(units: list[dict[str, Any]], **overrides: Any) -> Any:
    payload: dict[str, Any] = {
        "units": [UnitVerdict(**unit) for unit in units],
        "all_required_supported": all(unit.get("supported") for unit in units),
    }
    payload.update(overrides)
    return GroundingResult(**payload)


async def test_verify_draft_negative_verdicts_stay_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    async def _run_all_unavailable(*args: Any, **kwargs: Any) -> Any:
        return _grounding(
            [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": False,
                    "addresses_intent": False,
                }
            ],
            all_required_supported=False,
            unavailable_unit_ids=["u1"],
        )

    async def _run_incomplete(*args: Any, **kwargs: Any) -> Any:
        return _grounding(
            [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": True,
                    "addresses_intent": True,
                },
                {
                    "unit_id": "u2",
                    "scope": "book",
                    "supported": False,
                    "addresses_intent": True,
                },
            ],
            all_required_supported=False,
            unavailable_unit_ids=[],
        )

    async def _run_irrelevant(*args: Any, **kwargs: Any) -> Any:
        return _grounding(
            [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": True,
                    "addresses_intent": False,
                }
            ],
            all_required_supported=True,
            answer_relevant=False,
        )

    cases: tuple[tuple[Any, str], ...] = (
        (_run_all_unavailable, "verdict-missing"),
        (_run_incomplete, "verifier-incomplete-verdict"),
        (_run_irrelevant, "verdict-irrelevant"),
    )
    for replacement, expected_reason in cases:
        monkeypatch.setattr(turn_pipeline, "run_verifier", replacement)
        telemetry: dict[str, Any] = {}
        units, result, ok = await _verify_draft(
            "Первое предложение. Второе предложение.",
            [{"source_id": "s1"}],
            verifier_model=_StubModel(),
            telemetry=telemetry,
        )
        assert not ok
        assert units
        assert telemetry["answer_failure_cause"]["reason"] == expected_reason
        monkeypatch.undo()
