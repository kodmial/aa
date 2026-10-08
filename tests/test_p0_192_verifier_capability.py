"""P0 kodmial/aa#192 regression: verifier treats native structured output as optional.

Live evidence on exact main showed the simplified structured verifier request
never reaching a served verdict on the Space Bunny fallback path (14/14
``unavailable``, no ``aa-verifier-v2`` served identity) while ordinary text
calls on the same fallback serve. These tests pin the capability-compatible
repair at the adapter boundary (OpenCodeChatModel + scripted OpenCode client),
not only fake-model unit tests, so a provider path that never returns a
native structured verdict cannot look green:

- structured-capability failure falls back once to bounded plain-text JSON
  through the ordinary text path and still yields a strictly validated
  decision (same single concurrent per-unit round);
- provider 429 always propagates immediately and never triggers text fallback;
- invalid text JSON / missing keys / wrong types / unknown citations still
  fail closed (unknown envelope keys are dropped without being trusted;
  see kodmial/aa#217 recurrence 7);
- planner/retrieval/answer are untouched by the verifier text fallback.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from aa.conversation.model_adapter import OpenCodeChatModel, clear_primary_circuit
from aa.conversation.response_units import split_response_units
from aa.conversation.verifier import (
    parse_text_json_decision,
    run_verifier,
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
def _clear_circuit() -> Any:
    from aa.conversation.verifier import clear_verifier_capability_cache

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
    """Scripted OpenCode client at the transport boundary.

    ``structured_outcomes`` drive ``send_structured_message``;
    ``text_outcomes`` drive ``send_message``. An outcome may be a dict/str
    reply or a raised exception instance.
    """

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

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        **_: Any,
    ) -> str:
        self.text_calls += 1
        outcome = self.text_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)

    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        **_: Any,
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
            "addresses_intent": bool(supported),
        }
    )


async def test_structured_missing_falls_back_to_text_json_at_adapter_boundary() -> None:
    """A provider path with no native structured verdict still serves via text."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает. Тяга проходит спокойно.")
    assert len(units) == 2
    client = _AdapterScriptedClient(
        structured_outcomes=[
            OpenCodeDeterministicError("opencode structured output missing"),
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
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert [v.unit_id for v in result.units] == ["u1", "u2"]
    # Exactly one logical concurrent round: one structured attempt plus one
    # bounded text attempt per unit, no second verifier round.
    assert client.structured_calls == len(units)
    assert client.text_calls == len(units)


async def test_text_fallback_rejects_invalid_json_fail_closed() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    client = _AdapterScriptedClient(
        structured_outcomes=[OpenCodeDeterministicError("opencode structured output missing")],
        text_outcomes=["не json вообще"],
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-verifier-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    _, result, passed = await _verify_draft("Поддержка рядом помогает.", pack, verifier_model=model)
    assert passed is False
    assert result is None


async def test_text_fallback_rejects_unknown_citations_fail_closed() -> None:
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    bad = json.dumps(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["no-such-passage"],
            "addresses_intent": True,
        }
    )
    client = _AdapterScriptedClient(
        structured_outcomes=[OpenCodeDeterministicError("opencode structured output missing")],
        text_outcomes=[bad],
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-verifier-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    _, result, passed = await _verify_draft("Поддержка рядом помогает.", pack, verifier_model=model)
    assert passed is False
    assert result is None


async def test_structured_429_never_falls_back_to_text() -> None:
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


async def test_native_structured_success_uses_no_text_call() -> None:
    pack = [_pack_entry()]
    units = split_response_units("Понимаю. Поддержка рядом помогает.")
    client = _AdapterScriptedClient(
        structured_outcomes=[
            {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            },
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
            },
        ],
        text_outcomes=[],
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-verifier-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert client.text_calls == 0
    assert client.structured_calls == len(units)


def test_parse_text_json_decision_strict() -> None:
    pack_id = "chapter-3#exp0000"
    ok = parse_text_json_decision(
        '```json\n{"requires_book_evidence": true, "supported": true, '
        f'"evidence_passage_ids": ["{pack_id}"], "addresses_intent": true}}\n```'
    )
    assert ok["supported"] is True
    assert ok["evidence_passage_ids"] == [pack_id]
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision("просто текст без json")
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision('["not", "an", "object"]')
    # Envelope tolerance (kodmial/aa#217 recurrence 7): model-invented
    # keys are dropped without being trusted, so a fully-determined
    # verdict does not burn a second slow text round-trip. The unit id
    # stays bound by AA code and the aggregate stays AA-computed, so a
    # "unit_id" or "all_required_supported" key can never take effect;
    # grounding semantics are unchanged (verdict is a function of the
    # four known keys only).
    extra = parse_text_json_decision(
        json.dumps(
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
                "unit_id": "u1",
                "reasoning": "model prose habit",
            }
        )
    )
    assert extra == {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": [],
        "addresses_intent": True,
    }
    # Missing required keys and wrong value types still fail closed.
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(json.dumps({"supported": True, "evidence_passage_ids": []}))
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(
            json.dumps(
                {
                    "requires_book_evidence": True,
                    "supported": ["not-a-bool"],
                    "evidence_passage_ids": [],
                }
            )
        )
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(json.dumps({"comment": "nothing but envelope"}))


def test_planner_has_no_text_json_fallback() -> None:
    """Planner/retrieval/answer stay untouched by the verifier text fallback."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    planner = (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8")
    assert "parse_text_json_decision" not in planner
    assert "VERIFIER_TEXT_JSON_SUFFIX" not in planner


def test_repair_loop_wakes_issue_scheduler_explicitly() -> None:
    """Convergence automation gap: the repair publisher must wake the scheduler.

    Events emitted by GITHUB_TOKEN (repair issue create/update) do not
    reliably trigger downstream workflows, and cron alone left a P0 repair
    issue undispatched for roughly an hour. The self-proving workflow must
    dispatch the issue scheduler directly after publishing the repair issue.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "aa-self-proving-qualification.yml").read_text(
        encoding="utf-8"
    )
    assert "actions: write" in workflow
    assert "continuum-issue-scheduler.yml" in workflow
    assert "createWorkflowDispatch" in workflow


async def test_text_fallback_unsupported_verdict_stays_unsupported() -> None:
    """Grounding is not weakened: an unsupported text verdict blocks the draft."""
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    bad = json.dumps(
        {
            "requires_book_evidence": True,
            "supported": False,
            "evidence_passage_ids": [pack[0]["passage_id"]],
            "addresses_intent": False,
        }
    )
    client = _AdapterScriptedClient(
        structured_outcomes=[OpenCodeDeterministicError("opencode structured output missing")],
        text_outcomes=[bad],
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-verifier-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    units, result, passed = await _verify_draft(
        "Поддержка рядом помогает.", pack, verifier_model=model
    )
    assert passed is False
    assert result is not None
    assert result.all_required_supported is False
    assert units
