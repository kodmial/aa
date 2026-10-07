"""P0 kodmial/aa#200 regression: repair Gate C live-path-failed.

Live evidence on exact main d615df5 (run 37602056194) showed the
live-telegram-evidence lane failing with the concrete component signature:

- live-delivery-unsupported-out-of-book-timeout (one held-out turn never
  delivered within the per-turn wait),
- live-answer-no-generic-collapse (11 generic clarifications),
- live-answer-diversity (distinct replies below the diversity floor),
- live-delivery-sendmessage-observed (delivered count below 16 scenarios),

with stage outcomes planner ``ok`` 8, retrieval ``evidence-ready`` 8,
answer ``clarification`` 9 vs ``narrowed-supported`` 1, and verifier
``unavailable`` 9 vs ``unsupported`` 1 while every agent served the
configured fallback on the same path. A single text validation retry
(#198, two bounded attempts) still left most fallback turns unavailable:
the weak fallback path needs a second bounded text retry (three attempts
total) before failing closed. Grounding stays strict (every attempt is
Pydantic-validated); provider errors including 429 never retry and
propagate for runner retire/restart. Product Contract #110 stays
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
)
from aa.conversation.verifier_schema import VerifierValidationError
from aa.opencode.client import SessionInfo
from aa.opencode.errors import OpenCodeRateLimitError

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


def _decision_json(passage_id: str, *, supported: bool = True) -> str:
    return json.dumps(
        {
            "requires_book_evidence": True,
            "supported": supported,
            "evidence_passage_ids": [passage_id] if supported else [],
        }
    )


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


async def test_two_invalid_then_valid_serves_on_third_attempt() -> None:
    """Two non-compliant replies get a third bounded attempt and serve."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает.")
    assert len(units) == 1
    client = _AdapterScriptedClient(
        text_outcomes=[
            "Конечно, постараюсь помочь.",
            "просто текст без json",
            _decision_json(pack[0]["passage_id"]),
        ],
    )
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-verifier-v2",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )
    from aa.conversation import verifier as verifier_module

    verifier_module.mark_structured_unavailable(model)
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert client.text_calls == 3


async def test_three_invalid_text_fails_closed_bounded() -> None:
    """Three non-compliant replies fail closed with exactly three calls."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает.")
    client = _AdapterScriptedClient(
        text_outcomes=["no json here", "still no json", "again no json"],
    )

    class _TextOnlyModel:
        agent = "aa-verifier-v2"
        primary_model = PRIMARY
        fallback_model = FALLBACK

        def __init__(self) -> None:
            self.text_calls = 0

        async def ainvoke_structured(  # pragma: no cover - never used
            self,
            prompt: str,
            *,
            system: str = "",
            schema: object = None,
            retry_count: int = 1,
        ) -> dict[str, object]:
            raise AssertionError("unreachable")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            self.text_calls += 1
            return str(client.text_outcomes.pop(0))

    model = _TextOnlyModel()
    from aa.conversation import verifier as verifier_module

    verifier_module.mark_structured_unavailable(model)
    with pytest.raises(VerifierValidationError):
        await run_verifier(units, pack, model=model)
    assert model.text_calls == 3


async def test_text_429_on_retry_never_retries_further() -> None:
    """Provider 429 on a retry attempt propagates without further retries."""
    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает.")

    class _ValidationThenRatelimitModel:
        agent = "aa-verifier-v2"
        primary_model = PRIMARY
        fallback_model = FALLBACK

        def __init__(self) -> None:
            self.text_calls = 0

        async def ainvoke_structured(  # pragma: no cover - never used
            self,
            prompt: str,
            *,
            system: str = "",
            schema: object = None,
            retry_count: int = 1,
        ) -> dict[str, object]:
            raise AssertionError("unreachable")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            self.text_calls += 1
            if self.text_calls == 1:
                return "просто текст без json"
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    model = _ValidationThenRatelimitModel()
    from aa.conversation import verifier as verifier_module

    verifier_module.mark_structured_unavailable(model)
    with pytest.raises(OpenCodeRateLimitError):
        await run_verifier(units, pack, model=model)
    assert model.text_calls == 2
