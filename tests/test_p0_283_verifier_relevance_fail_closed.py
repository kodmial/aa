"""P0 kodmial/aa#283: fail-closed production verifier relevance.

Proves the production semantic-relevance repair with invented fixture
text only (no canonical book text, no exact live-question branches):

- an omitted, null, or wrong-type ``addresses_intent`` fails closed on
  both the provider-native and the text-JSON paths, even when the unit
  is book-supported with valid passage ids and checksums;
- a book-supported but off-topic unit (explicit ``addresses_intent``
  false with valid citations) is not relevant, cannot verify, and cannot
  count as a successful qualified answer;
- mixed glue/book turns still fail when the only book unit is off-topic
  and still pass when a relevant book unit plus glue is present;
- a held-out synthetic Russian 2-turn follow-up proves end-to-end
  behavior: a recycled broadly-topical passage on an elliptical
  mechanism follow-up fails relevance and never serves as qualified,
  while genuinely relevant supported guidance (or an honest limitation)
  is served.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from aa.conversation.response_units import split_response_units
from aa.conversation.verifier import (
    clear_verifier_capability_cache,
    coerce_single_verdict,
    parse_text_json_decision,
    run_verifier,
)
from aa.conversation.verifier_schema import (
    VerifierValidationError,
    validate_grounding_result,
    validate_unit_decision,
)


@pytest.fixture(autouse=True)
def _clear_verifier_cache_between_tests() -> Any:
    clear_verifier_capability_cache()
    yield
    clear_verifier_capability_cache()


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Фиктивная поддержка рядом. Тяга проходит, если обратиться за помощью.",
) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class _ScriptedStructured:
    """Provider-native structured fake returning scripted decisions."""

    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self._decisions = list(decisions)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        if not self._decisions:
            raise AssertionError("verifier called more times than scripted")
        return dict(self._decisions.pop(0))


class _StructuredDownTextUp:
    """Native structured channel down; bounded text JSON serves instead."""

    def __init__(self, texts: list[str]) -> None:
        from aa.opencode.errors import OpenCodeDeterministicError

        self._texts = list(texts)
        self._error = OpenCodeDeterministicError("opencode structured output missing")
        self.text_calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        raise self._error

    async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
        _ = (prompt, system)
        self.text_calls += 1
        if not self._texts:
            raise AssertionError("text verifier called more times than scripted")
        return self._texts.pop(0)


def _book_decision(passage_id: str, *, addresses: Any) -> dict[str, Any]:
    return {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": [passage_id],
        "addresses_intent": addresses,
    }


def test_omitted_relevance_fails_closed_native_and_text() -> None:
    pack = [_pack_entry()]
    omitted = {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": [pack[0]["passage_id"]],
    }
    with pytest.raises(VerifierValidationError):
        validate_unit_decision(dict(omitted))
    with pytest.raises(VerifierValidationError):
        coerce_single_verdict(dict(omitted), unit_id="u1")
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(json.dumps(dict(omitted)))


def test_null_relevance_fails_closed() -> None:
    pack = [_pack_entry()]
    payload = _book_decision(pack[0]["passage_id"], addresses=None)
    with pytest.raises(VerifierValidationError):
        validate_unit_decision(dict(payload))
    with pytest.raises(VerifierValidationError):
        coerce_single_verdict(dict(payload), unit_id="u1")
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(json.dumps(dict(payload)))


@pytest.mark.parametrize("bad", ["true", "True", "1", 1, 0, 1.0, [], {}])
def test_wrong_type_relevance_fails_closed(bad: Any) -> None:
    pack = [_pack_entry()]
    payload = _book_decision(pack[0]["passage_id"], addresses=bad)
    with pytest.raises(VerifierValidationError):
        validate_unit_decision(dict(payload))
    with pytest.raises(VerifierValidationError):
        coerce_single_verdict(dict(payload), unit_id="u1")
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(json.dumps(dict(payload)))


async def test_structured_omitted_relevance_never_becomes_pass() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает пережить тягу спокойно."
    model = _ScriptedStructured(
        [
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
            }
        ]
    )
    # A strict structured mock without a text path surfaces the
    # fail-closed validation error instead of silently passing.
    with pytest.raises(VerifierValidationError):
        await run_verifier(split_response_units(draft), pack, model=model)
    _, result, passed = await _verify_draft(draft, pack, verifier_model=model)
    assert passed is False
    assert result is None


async def test_text_omitted_relevance_never_becomes_pass() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает пережить тягу спокойно."
    bad = json.dumps(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": [pack[0]["passage_id"]],
        }
    )
    model = _StructuredDownTextUp([bad, bad])
    _, result, passed = await _verify_draft(draft, pack, verifier_model=model)
    assert passed is False
    assert result is None


async def test_supported_but_off_topic_book_unit_fails_relevance() -> None:
    from aa.conversation.answer_adequacy import (
        FAILURE_IRRELEVANT_CITATION,
        assess_turn_adequacy,
    )
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает пережить тягу спокойно."
    model = _ScriptedStructured([_book_decision(pack[0]["passage_id"], addresses=False)])
    units, result, passed = await _verify_draft(
        draft,
        pack,
        verifier_model=model,
        resolved_intent="synthetic resolved mechanism intent",
        user_message="synthetic follow-up",
    )
    assert passed is False
    assert result is not None
    assert result.all_required_supported is True
    assert result.answer_relevant is False
    assert result.relevance_category == "irrelevant-citation"
    assessment = assess_turn_adequacy(
        user_message="synthetic follow-up",
        reply=draft,
        evidence_pack=pack,
        grounding_result={
            "units": [
                {
                    "unit_id": verdict.unit_id,
                    "scope": verdict.scope,
                    "supported": verdict.supported,
                    "evidence_passage_ids": list(verdict.evidence_passage_ids),
                    "addresses_intent": verdict.addresses_intent,
                }
                for verdict in result.units
            ],
            "all_required_supported": result.all_required_supported,
            "answer_relevant": result.answer_relevant,
        },
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        planner_query_count=2,
    )
    assert assessment.verdict == "fail"
    assert assessment.failure_category == FAILURE_IRRELEVANT_CITATION
    assert assessment.answers_request is False
    assert units


async def test_text_path_off_topic_book_unit_fails_relevance() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает пережить тягу спокойно."
    bad = json.dumps(_book_decision(pack[0]["passage_id"], addresses=False))
    model = _StructuredDownTextUp([bad])
    _, result, passed = await _verify_draft(
        draft,
        pack,
        verifier_model=model,
        resolved_intent="synthetic resolved mechanism intent",
        user_message="synthetic follow-up",
    )
    assert passed is False
    assert result is not None
    assert result.answer_relevant is False


async def test_mixed_glue_and_off_topic_book_still_fails() -> None:
    pack = [_pack_entry()]
    units = split_response_units("Понимаю. Поддержка рядом помогает пережить тягу.")
    assert len(units) == 2
    model = _ScriptedStructured(
        [
            {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            },
            _book_decision(pack[0]["passage_id"], addresses=False),
        ]
    )
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert result.answer_relevant is False
    assert result.relevance_category == "irrelevant-citation"


async def test_mixed_glue_and_relevant_book_passes() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Понимаю. Поддержка рядом помогает пережить тягу."
    model = _ScriptedStructured(
        [
            {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            },
            _book_decision(pack[0]["passage_id"], addresses=True),
        ]
    )
    _, result, passed = await _verify_draft(
        draft,
        pack,
        verifier_model=model,
        resolved_intent="synthetic resolved intent",
        user_message="synthetic request",
    )
    assert passed is True
    assert result is not None
    assert result.all_required_supported is True
    assert result.answer_relevant is True


async def test_valid_relevant_book_answer_passes() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает пережить тягу спокойно."
    model = _ScriptedStructured([_book_decision(pack[0]["passage_id"], addresses=True)])
    _, result, passed = await _verify_draft(
        draft,
        pack,
        verifier_model=model,
        resolved_intent="synthetic resolved intent",
        user_message="synthetic request",
    )
    assert passed is True
    assert result is not None
    assert result.answer_relevant is True
    validated = validate_grounding_result(
        {
            "units": [
                {
                    "unit_id": verdict.unit_id,
                    "scope": verdict.scope,
                    "supported": verdict.supported,
                    "evidence_passage_ids": list(verdict.evidence_passage_ids),
                    "addresses_intent": verdict.addresses_intent,
                }
                for verdict in result.units
            ],
            "all_required_supported": result.all_required_supported,
        },
        expected_unit_ids=[verdict.unit_id for verdict in result.units],
    )
    assert validated.answer_relevant is True


async def test_held_out_synthetic_two_turn_followup_relevance() -> None:
    """Synthetic general question plus elliptical mechanism follow-up.

    Turn one grounds a broadly topical passage for a general recovery
    question. Turn two resolves an elliptical follow-up to the concrete
    mechanism: recycling the broadly topical passage is supported but
    off-topic and must fail closed (never qualified), while the concrete
    mechanism passage with explicit relevance passes.
    """

    from aa.conversation.turn_pipeline import run_v2_answer_turn

    broad_text = "Фиктивная общая поддержка рядом помогает держаться."
    mechanism_text = "Фиктивный разбор вечернего распорядка помогает пережить тягу."
    broad = _pack_entry(passage_id="chapter-3#exp0000", text=broad_text)
    mechanism = _pack_entry(passage_id="chapter-3#exp0001", text=mechanism_text)

    class _Answer:
        def __init__(self, drafts: list[str]) -> None:
            self._drafts = list(drafts)
            self.calls = 0

        async def ainvoke(self, messages: Any) -> str:
            _ = messages
            self.calls += 1
            return self._drafts.pop(0)

    first_draft = "Общая поддержка рядом помогает держаться."
    first = await run_v2_answer_turn(
        user_message="синтетический общий вопрос про восстановление",
        summary="",
        recent=[],
        evidence_pack=[broad, mechanism],
        answer_model=_Answer([first_draft]),
        verifier_model=_ScriptedStructured([_book_decision(broad["passage_id"], addresses=True)]),
        initial_query_count=2,
        resolved_intent="синтетический общий запрос про восстановление",
    )
    assert first["text"] == first_draft
    assert first["telemetry"]["adequacy_verdict"] == "pass"
    assert first["telemetry"]["qualified"] is True

    # Elliptical follow-up recycles the broadly topical passage. The unit
    # is supported by that passage but does not address the resolved
    # concrete-mechanism intent, so it must fail and never count as
    # qualified even though citations and checksums are valid.
    recycled_draft = "Общая поддержка рядом помогает держаться."
    import pytest as _pt283

    from aa.conversation.failures import TurnFailed as _TF283

    with _pt283.raises(_TF283) as _exc283:
        await run_v2_answer_turn(
            user_message="синтетическое уточнение про вечерний распорядок",
            summary="",
            recent=[],
            evidence_pack=[broad, mechanism],
            answer_model=_Answer([recycled_draft, recycled_draft]),
            verifier_model=_ScriptedStructured(
                [
                    _book_decision(broad["passage_id"], addresses=False),
                    {
                        "requires_book_evidence": False,
                        "supported": True,
                        "evidence_passage_ids": [],
                        "addresses_intent": True,
                    },
                ]
            ),
            initial_query_count=2,
            resolved_intent="синтетическое уточнение про конкретный вечерний распорядок",
        )
    recycled: dict[str, Any] = {"text": "", "telemetry": dict(_exc283.value.telemetry)}
    assert _exc283.value.category in ("adequacy-failed", "clarification-unavailable")
    assert recycled["telemetry"]["adequacy_verdict"] == "fail"
    assert recycled["telemetry"]["qualified"] is False

    # The same follow-up with the concrete mechanism passage and explicit
    # relevance passes end to end.
    mechanism_draft = "Вечерний распорядок помогает пережить тягу."
    mechanism_pack = [_pack_entry(passage_id="chapter-3#exp0001", text=mechanism_text)]
    served = await run_v2_answer_turn(
        user_message="синтетическое уточнение про вечерний распорядок",
        summary="",
        recent=[],
        evidence_pack=mechanism_pack,
        answer_model=_Answer([mechanism_draft]),
        verifier_model=_ScriptedStructured(
            [_book_decision(mechanism_pack[0]["passage_id"], addresses=True)]
        ),
        initial_query_count=2,
        resolved_intent="синтетическое уточнение про конкретный вечерний распорядок",
    )
    assert served["text"] == mechanism_draft
    assert served["telemetry"]["qualified"] is True
    # Privacy-safe telemetry carries booleans/counts/trace ids only.
    telemetry_text = json.dumps(served["telemetry"], ensure_ascii=False)
    assert broad_text not in telemetry_text
    assert mechanism_text not in telemetry_text
    assert mechanism_draft not in telemetry_text
    assert served["telemetry"]["turn_trace_id"]


def test_no_exact_question_oracle_in_production() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    verifier = (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8")
    schema = (root / "src" / "aa" / "conversation" / "verifier_schema.py").read_text(
        encoding="utf-8"
    )
    for probe in (
        "синтетический общий вопрос",
        "вечерний распорядок",
        "support implies relevance",
        "had_explicit_relevance",
    ):
        assert probe not in verifier
        assert probe not in schema
