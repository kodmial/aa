"""P0 kodmial/aa#190 regression: per-unit boolean verifier transport.

Proves the authoritative transport repair through invented fixture text
only (no canonical book text, no exact live-question special cases):

- native schema carries booleans, never a free-form scope string;
- the model never copies unit ids and never computes the aggregate;
- application code binds unit_id, derives scope deterministically, and
  computes all_required_supported;
- unknown/missing citations still fail closed;
- provider 429 propagates with a bounded single concurrent round;
- exactly one concurrent verifier round runs (no batch-first round).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import pytest

from aa.conversation.response_units import split_response_units
from aa.conversation.v2_prompts import load_verifier_system_v2
from aa.conversation.verifier import (
    build_single_unit_text,
    coerce_single_verdict,
    run_verifier,
)
from aa.conversation.verifier_schema import (
    VERIFIER_MAX_ATTEMPTS,
    VerifierValidationError,
    validate_unit_decision,
    verifier_single_json_schema,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONVERSATION_PKG = ROOT / "src" / "aa" / "conversation"


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
        "char_end": 120,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class _ScriptedVerifier:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self._decisions = list(decisions)
        self.calls = 0
        self.schemas: list[dict[str, object]] = []

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, retry_count)
        self.calls += 1
        self.schemas.append(dict(schema))
        if not self._decisions:
            raise AssertionError("verifier called more times than scripted")
        return dict(self._decisions.pop(0))


def test_transport_schema_has_booleans_not_scope() -> None:
    schema = verifier_single_json_schema()
    dumped = json.dumps(schema)
    assert "$ref" not in dumped
    assert '"enum"' not in dumped
    props = schema["properties"]
    assert isinstance(props, dict)
    assert set(props) == {"requires_book_evidence", "supported", "evidence_passage_ids"}
    assert props["requires_book_evidence"] == {"type": "boolean"}
    assert props["supported"] == {"type": "boolean"}
    assert schema["required"] == [
        "requires_book_evidence",
        "supported",
        "evidence_passage_ids",
    ]
    assert "scope" not in dumped
    assert "unit_id" not in dumped
    assert "all_required_supported" not in dumped
    assert "book" not in dumped or "requires_book_evidence" in dumped
    # Scope-shaped payloads are rejected at the transport boundary.
    with pytest.raises(VerifierValidationError):
        validate_unit_decision({"scope": "book", "supported": True, "evidence_passage_ids": []})
    decision = validate_unit_decision(
        {
            "requires_book_evidence": False,
            "supported": True,
            "evidence_passage_ids": [],
        }
    )
    assert decision.requires_book_evidence is False
    assert VERIFIER_MAX_ATTEMPTS == 1


def test_no_unit_id_copying_and_no_model_aggregate() -> None:
    units = split_response_units("Понимаю. Тяга проходит спокойно.")
    pack = [_pack_entry()]
    text = build_single_unit_text(unit=units[0], passages=pack)
    assert "unit_id" not in text
    assert "u1" not in text
    assert "all_required_supported" not in text
    assert "scope" not in text
    # Application code binds the id; the model output carries none.
    verdict = coerce_single_verdict(
        {
            "requires_book_evidence": False,
            "supported": True,
            "evidence_passage_ids": [],
        },
        unit_id="u7",
    )
    assert verdict.unit_id == "u7"
    assert verdict.scope != "book"
    # Aggregate is computed in code, never accepted from the model.
    with pytest.raises(VerifierValidationError):
        validate_unit_decision(
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [],
                "all_required_supported": True,
            }
        )


def test_scope_derived_deterministically_in_code() -> None:
    bookish = coerce_single_verdict(
        {"requires_book_evidence": True, "supported": True, "evidence_passage_ids": ["p1"]},
        unit_id="u1",
    )
    glue = coerce_single_verdict(
        {"requires_book_evidence": False, "supported": True, "evidence_passage_ids": []},
        unit_id="u2",
    )
    assert bookish.scope == "book"
    assert glue.scope != "book"


async def test_unknown_and_missing_citations_fail_closed() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает."
    missing = _ScriptedVerifier(
        [{"requires_book_evidence": True, "supported": True, "evidence_passage_ids": []}]
    )
    _, result, passed = await _verify_draft(draft, pack, verifier_model=missing)
    assert passed is False
    assert result is None

    unknown = _ScriptedVerifier(
        [
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["no-such-passage"],
            }
        ]
    )
    _, result2, passed2 = await _verify_draft(draft, pack, verifier_model=unknown)
    assert passed2 is False
    assert result2 is None


class _PartiallyUnavailableVerifier:
    """One grounded unit succeeds while another transport call fails."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        if "Поддержка рядом помогает." in prompt:
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["p1"],
            }
        raise RuntimeError("fixture verifier transport failure")


async def test_partial_verifier_failure_preserves_verified_units() -> None:
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает. Ещё одна мысль.")
    result = await run_verifier(units, pack, model=_PartiallyUnavailableVerifier())
    assert result.all_required_supported is False
    assert result.unavailable_unit_ids == ["u2"]
    assert result.units[0].unit_id == "u1"
    assert result.units[0].supported is True
    assert result.units[0].scope == "book"
    assert result.units[1].unit_id == "u2"
    assert result.units[1].supported is False


async def test_partial_verifier_failure_narrows_instead_of_generic_collapse() -> None:
    from aa.conversation.turn_pipeline import (
        NATURAL_CLARIFICATION_REPLY,
        run_v2_answer_turn,
    )

    class _Answer:
        async def ainvoke(self, messages: Any) -> str:
            _ = messages
            return "Поддержка рядом помогает. Ещё одна мысль."

    outcome = await run_v2_answer_turn(
        user_message="что делать?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_Answer(),
        verifier_model=_PartiallyUnavailableVerifier(),
        initial_query_count=1,
    )
    assert outcome["text"] == "Поддержка рядом помогает."
    assert outcome["text"] != NATURAL_CLARIFICATION_REPLY
    assert outcome["telemetry"]["verifier_outcome"] == "partial-unavailable"
    assert outcome["telemetry"]["verifier_unavailable_units"] == 1
    assert outcome["telemetry"]["repair_rounds"] == 0


async def test_partial_verifier_failure_does_not_serve_glue_only_for_substantive_turn() -> None:
    from aa.conversation.turn_pipeline import NATURAL_CLARIFICATION_REPLY, run_v2_answer_turn

    class _Answer:
        async def ainvoke(self, messages: Any) -> str:
            _ = messages
            return "Понимаю. Содержательная мысль."

    class _GlueOnlyVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            if "Понимаю." in prompt:
                return {
                    "requires_book_evidence": False,
                    "supported": True,
                    "evidence_passage_ids": [],
                }
            raise RuntimeError("fixture verifier transport failure")

    outcome = await run_v2_answer_turn(
        user_message="что делать?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_Answer(),
        verifier_model=_GlueOnlyVerifier(),
        initial_query_count=1,
    )
    assert outcome["text"] == NATURAL_CLARIFICATION_REPLY
    assert outcome["telemetry"]["verifier_outcome"] == "partial-unavailable"


async def test_provider_429_propagates_with_bounded_round() -> None:
    from aa.opencode.errors import OpenCodeRateLimitError

    pack = [_pack_entry()]
    units = split_response_units("Понимаю. Поддержка рядом помогает.")

    class _Always429:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    model = _Always429()
    with pytest.raises(OpenCodeRateLimitError):
        await run_verifier(units, pack, model=model)
    assert model.calls == len(units)


async def test_one_concurrent_verifier_round_only() -> None:
    pack = [_pack_entry()]
    units = split_response_units("Понимаю. Поддержка рядом помогает.")
    assert len(units) == 2
    model = _ScriptedVerifier(
        [
            {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
            },
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
            },
        ]
    )
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert [v.unit_id for v in result.units] == ["u1", "u2"]
    # Exactly one request per unit, no batch-first round.
    assert model.calls == len(units)
    for schema in model.schemas:
        assert "units" not in str(schema)
        assert "scope" not in str(schema)


def test_no_exact_live_question_special_cases() -> None:
    system = load_verifier_system_v2()
    units = split_response_units("Понимаю. Тяга проходит.")
    user_text = build_single_unit_text(unit=units[0], passages=[_pack_entry()])
    for probe in (
        "тянет выпить",
        "ссора",
        "акции",
        "покончить",
        "37422302821",
        "37571901838",
        "2d4ba51d7c94",
    ):
        assert probe not in system
        assert probe not in user_text
    # Semantic rule is preserved without scope strings.
    assert "requires_book_evidence" in system
    assert "general offer" in system.casefold()
    assert "product_meta" not in system
    assert "conversation_glue" not in system
    for name in ("verifier.py", "verifier_schema.py"):
        source = (CONVERSATION_PKG / name).read_text(encoding="utf-8")
        assert "37571901838" not in source
        assert "2d4ba51d7c94" not in source


def test_verifier_source_has_no_batch_round() -> None:
    source = (CONVERSATION_PKG / "verifier.py").read_text(encoding="utf-8")
    assert "build_verifier_user_text" not in source
    assert "coerce_grounding_result" not in source
    assert "verifier_json_schema" not in source
    assert "_remap_units_by_order" not in source
    assert "run_verifier" in source
    assert "ainvoke_structured" in source
