"""Qualification #530: per-unit quarantine of deterministic gate failures.

A model-side citation or quotation mistake in one response unit must not
discard independently verified sibling units. The strict gates stay
unchanged for the rejected unit (it is never served), while the kept
subset is re-certified as one contiguous candidate. A single rejected
unit still fails the whole verification closed.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from aa.conversation.response_units import split_response_units
from aa.conversation.turn_pipeline import _verify_draft
from aa.conversation.verifier import run_batched_verifier, run_verifier
from aa.conversation.verifier_schema import VerifierValidationError


def _pack_entry(text: str, *, passage_id: str = "chapter-3#exp0000") -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "child_chunk_ids": [passage_id],
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


def _book_decision(passage_id: str) -> dict[str, Any]:
    return {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": [passage_id],
        "addresses_intent": True,
        "claim_origin": "book_claim",
    }


def _glue_decision() -> dict[str, Any]:
    return {
        "requires_book_evidence": False,
        "supported": True,
        "evidence_passage_ids": [],
        "addresses_intent": True,
        "claim_origin": "conversation_glue",
    }


class _ScriptedBatchedVerifier:
    supports_batched_verifier = True

    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self._decisions = [dict(item) for item in decisions]
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        return {"verdicts": [dict(item) for item in self._decisions]}


class _PromptAwarePerUnitVerifier:
    def __init__(self, decision: dict[str, Any]) -> None:
        self._decision = dict(decision)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        return dict(self._decision)


def _unit_from_prompt(prompt: str) -> str:
    import re

    match = re.search(r"<response_unit>(.*?)</response_unit>", prompt, re.S)
    return match.group(1).strip() if match else prompt


class _UnitTextVerifier:
    """Per-unit verifier that selects a decision by response-unit text."""

    def __init__(self, decisions: dict[str, dict[str, Any]]) -> None:
        self._decisions = {key: dict(value) for key, value in decisions.items()}
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        self.calls += 1
        return dict(self._decisions[_unit_from_prompt(prompt)])


class _StaticAnswer:
    def __init__(self, text: str) -> None:
        self._text = text

    async def ainvoke(self, messages: Any) -> Any:
        from langchain_core.messages import AIMessage

        _ = messages
        return AIMessage(content=self._text)


BOOK_TEXT = "Фиктивная поддержка рядом. Тяга проходит, если обратиться за помощью."
BAD_QUOTE_DRAFT = (
    "Вот точные слова: «Фиктивная поддержка рядом помогает». "
    "Тяга проходит, если обратиться за помощью."
)
EXACT_QUOTE_SIBLING_DRAFT = (
    "Вот неточные слова: «Фиктивная поддержка рядом помогает». "
    "Вот точные слова: «Тяга проходит, если обратиться за помощью»."
)
CROSS_UNIT_QUOTE_DRAFT = (
    "«Фиктивная поддержка рядом. Тяга проходит, если обратиться за помощью с чужой помощью»."
)
GLUE_QUOTE_DRAFT = "Понимаю, это «качели». Тяга проходит, если обратиться за помощью."
USER_MESSAGE = "К вечеру очень тянет выпить, как с этим обходиться?"


@pytest.fixture(autouse=True)
def _enable_turn_time_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "true")


async def test_batched_verifier_quarantines_invented_quote_unit() -> None:
    """The invented quote unit is blocked; the exact sibling stays verified."""
    pack = [_pack_entry(BOOK_TEXT)]
    units = split_response_units(BAD_QUOTE_DRAFT)
    assert len(units) == 2
    model = _ScriptedBatchedVerifier([_book_decision(pack[0]["passage_id"])] * 2)

    result = await run_batched_verifier(
        units,
        pack,
        model=model,
        user_message=USER_MESSAGE,
        answer_text=BAD_QUOTE_DRAFT,
    )

    assert [item.unit_id for item in result.units] == ["u1", "u2"]
    assert result.units[0].supported is False
    assert result.units[0].evidence_passage_ids == []
    assert result.units[1].supported is True
    assert result.all_required_supported is False


async def test_batched_verifier_keeps_exact_quote_sibling() -> None:
    """A verified exact quotation survives next to an invented quote."""
    pack = [_pack_entry(BOOK_TEXT)]
    units = split_response_units(EXACT_QUOTE_SIBLING_DRAFT)
    assert len(units) == 2
    model = _ScriptedBatchedVerifier([_book_decision(pack[0]["passage_id"])] * 2)

    result = await run_batched_verifier(
        units,
        pack,
        model=model,
        user_message=USER_MESSAGE,
        answer_text=EXACT_QUOTE_SIBLING_DRAFT,
    )

    assert result.units[0].supported is False
    assert result.units[1].supported is True
    assert result.units[1].evidence_passage_ids == [pack[0]["passage_id"]]


async def test_cross_unit_quotation_still_fails_closed() -> None:
    """A quote spanning two units never borrows support from either side."""
    pack = [_pack_entry(BOOK_TEXT)]
    units = split_response_units(CROSS_UNIT_QUOTE_DRAFT)
    assert len(units) == 2
    model = _ScriptedBatchedVerifier([_book_decision(pack[0]["passage_id"])] * 2)

    with pytest.raises(VerifierValidationError):
        await run_batched_verifier(
            units,
            pack,
            model=model,
            user_message=USER_MESSAGE,
            answer_text=CROSS_UNIT_QUOTE_DRAFT,
        )
    assert model.calls == 1


async def test_batched_verifier_quarantines_unattributed_glue_quote() -> None:
    """An unprovable emphasis quote fails only its own glue unit."""
    pack = [_pack_entry(BOOK_TEXT)]
    units = split_response_units(GLUE_QUOTE_DRAFT)
    assert len(units) == 2
    model = _ScriptedBatchedVerifier([_glue_decision(), _book_decision(pack[0]["passage_id"])])

    result = await run_batched_verifier(
        units,
        pack,
        model=model,
        user_message=USER_MESSAGE,
        answer_text=GLUE_QUOTE_DRAFT,
    )

    assert result.units[0].supported is False
    assert result.units[1].supported is True
    assert result.all_required_supported is False


async def test_verify_draft_preserves_verified_sibling_instead_of_none() -> None:
    """A per-unit gate rejection must not collapse the whole draft to None."""
    pack = [_pack_entry(BOOK_TEXT)]
    model = _ScriptedBatchedVerifier([_book_decision(pack[0]["passage_id"])] * 2)

    units, result, passed = await _verify_draft(
        BAD_QUOTE_DRAFT,
        pack,
        verifier_model=model,
        user_message=USER_MESSAGE,
    )

    assert len(units) == 2
    assert result is not None
    assert passed is False
    assert result.units[0].supported is False
    assert result.units[1].supported is True


async def test_turn_serves_verified_book_sibling_without_glue_quote() -> None:
    """The delivery boundary serves the verified sibling, not the rejected glue."""
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    book_unit = "Тяга проходит, если обратиться за помощью."
    glue_unit = "Понимаю, это «качели»."
    draft = f"{glue_unit} {book_unit}"
    pack = [_pack_entry(BOOK_TEXT)]
    verifier = _UnitTextVerifier(
        {
            glue_unit: _glue_decision(),
            book_unit: _book_decision(pack[0]["passage_id"]),
        }
    )

    outcome = await run_v2_answer_turn(
        user_message=USER_MESSAGE,
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=_StaticAnswer(draft),
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent=USER_MESSAGE,
    )

    assert outcome["text"] == book_unit
    assert "«качели»" not in outcome["text"]
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "narrowed-adequacy"
    assert telemetry["verifier_outcome"] == "passed"
    assert telemetry["qualified"] is True


async def test_per_unit_verifier_quarantines_invented_quote_unit() -> None:
    """The same per-unit quarantine applies to the non-batched transport."""
    pack = [_pack_entry(BOOK_TEXT)]
    units = split_response_units(BAD_QUOTE_DRAFT)
    model = _PromptAwarePerUnitVerifier(_book_decision(pack[0]["passage_id"]))

    result = await run_verifier(
        units,
        pack,
        model=model,
        user_message=USER_MESSAGE,
        answer_text=BAD_QUOTE_DRAFT,
    )

    assert model.calls == 2
    assert result.units[0].supported is False
    assert result.units[1].supported is True
    assert result.all_required_supported is False


async def test_single_rejected_unit_still_fails_closed() -> None:
    """With no independently verified unit left, the strict error propagates."""
    pack = [_pack_entry(BOOK_TEXT)]
    draft = "Вот точные слова: «Фиктивная поддержка рядом помогает»."
    units = split_response_units(draft)
    assert len(units) == 1
    model = _ScriptedBatchedVerifier([_book_decision(pack[0]["passage_id"])])

    with pytest.raises(VerifierValidationError):
        await run_batched_verifier(
            units,
            pack,
            model=model,
            user_message=USER_MESSAGE,
            answer_text=draft,
        )


async def test_user_quote_in_kept_sibling_stays_exempt_after_quarantine() -> None:
    """A quarantined book unit must not strip the kept sibling's exemption.

    The delivery boundary re-certifies the original draft, where the
    quarantined unit's demoted verdict (supported false, citations
    cleared) fails the invented book quotation wholesale. The trusted
    user-quote exemption must still be rebuilt from the kept subset so
    the faithful user quotation does not count toward the book quota.
    """
    from aa.conversation.turn_pipeline import certified_user_span_texts
    from aa.conversation.verifier_schema import NON_BOOK_SCOPE, GroundingResult, UnitVerdict

    user_quote = "К вечеру очень тянет выпить"
    draft = f"Вот точные слова: «Фиктивная поддержка рядом помогает». Вы сказали: «{user_quote}»."
    units = split_response_units(draft)
    assert len(units) == 2
    result = GroundingResult(
        units=[
            UnitVerdict(
                unit_id="u1",
                scope="book",
                supported=False,
                evidence_passage_ids=[],
                addresses_intent=False,
                origin="book_claim",
            ),
            UnitVerdict(
                unit_id="u2",
                scope=NON_BOOK_SCOPE,
                supported=True,
                evidence_passage_ids=[],
                addresses_intent=True,
                origin="user_report",
            ),
        ],
        all_required_supported=False,
    )

    spans = certified_user_span_texts(
        draft=draft,
        units=units,
        result=result,
        pack=[_pack_entry(BOOK_TEXT)],
        recent=[],
        user_message=USER_MESSAGE,
    )

    assert user_quote in spans
    assert "Фиктивная поддержка рядом помогает" not in spans
