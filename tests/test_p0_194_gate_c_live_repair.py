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
    """Small-model text deviations parse; required keys/types stay strict.

    Envelope tolerance (kodmial/aa#217 recurrence 7): unknown keys are
    dropped without being trusted instead of failing the whole decision,
    so a fully-determined verdict does not burn a second slow text
    round-trip per unit. Missing required keys, wrong value types, and
    non-object payloads still fail closed; grounding semantics are
    unchanged (verdict is a function of the three known keys only, unit
    id bound by AA code, aggregate computed by AA code).
    """
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
    # Unknown envelope keys are dropped without being trusted; the
    # verdict still comes from the three known keys only.
    assert parse_text_json_decision(
        json.dumps(
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [],
                "reasoning": "extra key",
            }
        )
    ) == {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": [],
    }
    # Missing keys and wrong types still fail closed.
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(json.dumps({"supported": True, "evidence_passage_ids": []}))


def test_verifier_suffix_has_explicit_example() -> None:
    """The text fallback prompt carries an explicit single-object example."""
    from aa.conversation.verifier import VERIFIER_TEXT_JSON_SUFFIX

    assert "Example:" in VERIFIER_TEXT_JSON_SUFFIX
    assert "p1" in VERIFIER_TEXT_JSON_SUFFIX
    assert "No other text" in VERIFIER_TEXT_JSON_SUFFIX


def test_production_verifier_is_muse_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifier is pinned to Muse Spark and never falls through to Space Bunny."""
    from types import SimpleNamespace

    import aa.conversation.graph as graph_module
    import aa.conversation.model_adapter as adapter_module
    from aa.config import DEFAULT_PRIMARY_MODEL
    from aa.conversation.graph_runtime import _ProductionGraphRuntime

    class _DummyModel:
        def __init__(
            self,
            client: object,
            *,
            agent: str,
            primary_model: str,
            fallback_model: str = "",
            request_timeout: float = 120.0,
            transport_agent: str | None = None,
        ) -> None:
            self.client = client
            self.agent = agent
            self.primary_model = primary_model
            self.fallback_model = fallback_model
            self.request_timeout = request_timeout
            self.transport_agent = transport_agent

        @property
        def wire_agent(self) -> str:
            if self.transport_agent is None:
                return self.agent
            return self.transport_agent

        def with_agent(self, agent: str) -> _DummyModel:
            return _DummyModel(
                self.client,
                agent=agent,
                primary_model=self.primary_model,
                fallback_model=self.fallback_model,
                request_timeout=self.request_timeout,
            )

    monkeypatch.setattr(adapter_module, "OpenCodeChatModel", _DummyModel)
    monkeypatch.setattr(graph_module, "build_turn_graph", lambda **kwargs: kwargs)

    runtime = _ProductionGraphRuntime(
        client=object(),
        settings=SimpleNamespace(
            opencode_model="opencode/muse-spark-1.3-contributor-free",
            opencode_fallback_model="opencode/space-bunny-free",
        ),
        index=None,
    )
    wired = runtime._build_graph(object())
    planner = wired["planner_model"]
    verifier = wired["verifier_model"]

    assert planner.fallback_model == "opencode/space-bunny-free"
    assert verifier.agent == "aa-verifier-v2"
    assert verifier.primary_model == DEFAULT_PRIMARY_MODEL
    assert verifier.primary_model == "opencode/muse-spark-1.3-contributor-free"
    assert verifier.fallback_model == ""
    # kodmial/aa#202: the logical audit identity stays aa-verifier-v2
    # while the transport selector is omitted on the wire.
    assert verifier.wire_agent == ""
