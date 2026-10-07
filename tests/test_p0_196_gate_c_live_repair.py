"""P0 kodmial/aa#196 regression: repair Gate C live-path-failed.

Live evidence on exact main 9b372aa (run 37593979925) showed the
live-telegram-evidence lane failing with the concrete component signature:

- live-answer-no-generic-collapse (14 generic clarifications of 14 turns),
- live-answer-diversity (distinct replies below the diversity floor),
- live-actual-served-model-identity (no aa-verifier-v2 served identity),
- live-text-max-over-budget (max 56.0s over the 30s hard budget),

with stage outcomes planner ``ok`` 14/14, retrieval ``evidence-ready`` 14/14,
answer ``clarification`` 14/14, and verifier ``unavailable`` 14/14 while the
configured fallback served planner/answer on the same path. The verifier must
therefore fall back once to the bounded text path on any non-429
provider-side structured failure (not only deterministic/transient subsets),
prefer text subsequently (TTL-bounded) so later turns do not burn two slow
provider round-trips per unit, and keep the configured model fallback so a
rejected primary cannot strand the gate. Product Contract #110 stays
immutable; no exact-question special cases.
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
    run_verifier,
    structured_text_fallback_preferred,
)
from aa.opencode.client import SessionInfo
from aa.opencode.errors import (
    OpenCodeNotReadyError,
    OpenCodeRateLimitError,
    OpenCodeStartupError,
    OpenCodeTransientError,
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


async def test_generic_provider_error_falls_back_to_text_once() -> None:
    """A generic provider-side structured failure still serves via text."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает.")
    assert len(units) == 1
    client = _AdapterScriptedClient(
        structured_outcomes=[OpenCodeNotReadyError("opencode runtime has not been started")],
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


async def test_startup_error_falls_back_to_text_once() -> None:
    """A startup-shaped structured failure still serves via text."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает.")
    client = _AdapterScriptedClient(
        structured_outcomes=[OpenCodeStartupError("failed to launch opencode serve")],
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


async def test_transient_marks_capability_to_bound_latency() -> None:
    """After a transient fallback the next round uses text directly."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает.")

    class _TransientThenTextModel:
        """Minimal verifier model: one transient structured failure, then text."""

        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0
            self.agent = "aa-verifier-v2"
            self.primary_model = PRIMARY
            self.fallback_model = FALLBACK

        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str = "",
            schema: dict[str, object] | None = None,
            retry_count: int = 1,
        ) -> dict[str, object]:
            self.structured_calls += 1
            raise OpenCodeTransientError("opencode request failed transiently")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            self.text_calls += 1
            return _decision_json(pack[0]["passage_id"])

    model = _TransientThenTextModel()
    first = await run_verifier(units, pack, model=model)
    assert first.all_required_supported is True
    assert structured_text_fallback_preferred(model) is True
    assert model.structured_calls == 1
    assert model.text_calls == 1
    second = await run_verifier(units, pack, model=model)
    assert second.all_required_supported is True
    assert model.structured_calls == 1
    assert model.text_calls == 2


async def test_structured_429_never_falls_back_to_text() -> None:
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
    assert structured_text_fallback_preferred(model) is False


async def test_invalid_structured_decision_prefers_text_afterwards() -> None:
    """After a schema-invalid structured object the next round uses text directly."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает.")

    class _InvalidThenTextModel:
        """Minimal verifier model: one invalid structured object, then text."""

        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0
            self.agent = "aa-verifier-v2"
            self.primary_model = PRIMARY
            self.fallback_model = FALLBACK

        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str = "",
            schema: dict[str, object] | None = None,
            retry_count: int = 1,
        ) -> dict[str, object]:
            self.structured_calls += 1
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [],
                "reasoning": "extra key",
            }

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            self.text_calls += 1
            return _decision_json(pack[0]["passage_id"])

    model = _InvalidThenTextModel()
    first = await run_verifier(units, pack, model=model)
    assert first.all_required_supported is True
    assert structured_text_fallback_preferred(model) is True
    assert model.structured_calls == 1
    assert model.text_calls == 1
    second = await run_verifier(units, pack, model=model)
    assert second.all_required_supported is True
    assert model.structured_calls == 1
    assert model.text_calls == 2
