"""P0 kodmial/aa#194 regression: repair Gate C live-path-failed.

Live evidence on exact main 61b649a showed the live-telegram-evidence lane
failing with the concrete component signature:

- live-answer-no-generic-collapse (13 generic clarifications of 14 turns),
- live-answer-diversity (distinct replies below the diversity floor),
- live-text-max-over-budget (max 40.0s over the 30s hard budget),

with stage outcomes verifier ``unavailable`` 12/14 and answer
``clarification`` 13/14 while planner/retrieval served on the same
Space Bunny fallback path. The native structured verifier channel is a
capability, not an assumption: after its first failure the process must
prefer the bounded text JSON path directly instead of burning two slow
provider round-trips per unit, structured validation failures must get one
bounded text chance before failing closed, and small-model text deviations
must parse tolerantly while Pydantic key/type strictness stays unchanged.

These tests pin that repair at the adapter boundary with scripted OpenCode
clients (never exact-question special cases, never Product Contract
weakening).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from aa.conversation.model_adapter import OpenCodeChatModel, clear_primary_circuit
from aa.conversation.response_units import split_response_units
from aa.conversation.verifier import (
    clear_verifier_capability_cache,
    parse_text_json_decision,
    run_verifier,
    structured_text_fallback_preferred,
)
from aa.conversation.verifier_schema import VerifierValidationError
from aa.opencode.client import SessionInfo
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeRateLimitError,
)

PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"


@pytest.fixture(autouse=True)
def _clear_state() -> Any:
    clear_primary_circuit()
    clear_verifier_capability_cache()
    yield
    clear_primary_circuit()
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
        "char_end": 120,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class _AdapterScriptedClient:
    """Scripted OpenCode client at the transport boundary."""

    def __init__(
        self,
        *,
        text_outcomes: list[object] | None = None,
        structured_outcomes: list[object] | None = None,
    ) -> None:
        self.text_outcomes = list(text_outcomes or [])
        self.structured_outcomes = list(structured_outcomes or [])
        self.text_calls = 0
        self.structured_calls = 0
        self._sessions = 0

    async def create_session(self, title: str = "") -> SessionInfo:
        self._sessions += 1
        return SessionInfo(id=f"ses_{self._sessions}", title=title)

    async def delete_session(self, session_id: str) -> bool:
        return True

    async def send_message(self, session_id: str, text: str, *, model: str = "", **_: Any) -> str:
        self.text_calls += 1
        outcome = self.text_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)

    async def send_structured_message(
        self, session_id: str, text: str, *, model: str = "", **_: Any
    ) -> dict[str, object]:
        self.structured_calls += 1
        outcome = self.structured_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, dict)
        return outcome


def _decision_json(passage_id: str, *, supported: bool = True) -> str:
    return json.dumps(
        {
            "requires_book_evidence": True,
            "supported": supported,
            "evidence_passage_ids": [passage_id] if supported else [],
        }
    )


async def test_structured_invalid_falls_back_to_text_once() -> None:
    """A structured object failing the decision schema still serves via text."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает.")
    assert len(units) == 1
    client = _AdapterScriptedClient(
        structured_outcomes=[
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [],
                "unit_id": "u1",
            }
        ],
        text_outcomes=[_decision_json(pack[0]["passage_id"])],
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-verifier-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert client.structured_calls == 1
    assert client.text_calls == 1


async def test_capability_cache_skips_structured_on_next_round() -> None:
    """After one structured-channel failure, the next round uses text directly."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает.")
    client = _AdapterScriptedClient(
        structured_outcomes=[
            OpenCodeDeterministicError("opencode structured output missing"),
        ],
        text_outcomes=[
            _decision_json(pack[0]["passage_id"]),
            _decision_json(pack[0]["passage_id"]),
        ],
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-verifier-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    first = await run_verifier(units, pack, model=model)
    assert first.all_required_supported is True
    assert structured_text_fallback_preferred() is True
    assert client.structured_calls == 1
    assert client.text_calls == 1
    # Second round in the same process must not burn another structured call.
    second = await run_verifier(units, pack, model=model)
    assert second.all_required_supported is True
    assert client.structured_calls == 1
    assert client.text_calls == 2


async def test_structured_429_never_marks_capability() -> None:
    """Provider 429 propagates immediately and never prefers the text path."""
    pack = [_pack_entry()]
    units = split_response_units("Понимаю. Поддержка рядом помогает.")
    client = _AdapterScriptedClient(
        structured_outcomes=[OpenCodeRateLimitError("opencode request rate-limited: http=429")]
        * len(units),
        text_outcomes=["must-never-be-consumed"] * len(units),
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-verifier-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    with pytest.raises(OpenCodeRateLimitError):
        await run_verifier(units, pack, model=model)
    assert client.text_calls == 0
    assert structured_text_fallback_preferred() is False


def test_tolerant_text_json_variants_still_strict_on_keys() -> None:
    """Small-model text deviations parse; extra keys still fail closed."""
    pack_id = "chapter-3#exp0000"
    trailing = (
        '{"requires_book_evidence": true, "supported": true, "evidence_passage_ids": ["p1"],}'
    )
    assert parse_text_json_decision(trailing)["evidence_passage_ids"] == ["p1"]
    single = "{'requires_book_evidence': True, 'supported': True, 'evidence_passage_ids': ['p1']}"
    assert parse_text_json_decision(single)["supported"] is True
    fenced = (
        "Here is the decision:\n```json\n"
        '{"requires_book_evidence": true, "supported": true, '
        f'"evidence_passage_ids": ["{pack_id}"]}}\n```'
    )
    assert parse_text_json_decision(fenced)["evidence_passage_ids"] == [pack_id]
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision("просто текст без json")
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(
            json.dumps(
                {
                    "requires_book_evidence": True,
                    "supported": True,
                    "evidence_passage_ids": [],
                    "reasoning": "extra key",
                }
            )
        )


def test_verifier_suffix_has_explicit_example() -> None:
    """The text fallback prompt carries an explicit single-object example."""
    from aa.conversation.verifier import VERIFIER_TEXT_JSON_SUFFIX

    assert "Example:" in VERIFIER_TEXT_JSON_SUFFIX
    assert "p1" in VERIFIER_TEXT_JSON_SUFFIX
    assert "No other text" in VERIFIER_TEXT_JSON_SUFFIX
