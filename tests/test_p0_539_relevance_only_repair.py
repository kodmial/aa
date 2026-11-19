"""Qualification #7 / kodmial/aa#539: relevance-only repair focus.

Failure class: a draft can be fully supported by the Evidence Pack yet
answer only the background of the request, so the verifier rejects the
whole turn as ``verdict-irrelevant``. Historical repair regeneration
reused the raw request, which asks for exactly the background the model
had already produced, and the same verified-but-irrelevant draft was
regenerated until the turn collapsed. The relevance-only repair focus
carries a generic practical-guidance instruction first and the live
request last; it is chosen from verifier fields only, never from text
heuristics or an exact qualification prompt.

These are synthetic fixtures for the failure class, never the literal
qualification prompt.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

_SYNTHETIC_USER_MESSAGE = "Синтетический вопрос о том, как справляться с вечерней привычкой?"
_SYNTHETIC_RESOLVED_INTENT = "синтетическое намерение справиться с вечерней привычкой"
_SYNTHETIC_BACKGROUND = "Изобретение описывает общее свойство вечерней привычки."
_SYNTHETIC_BACKGROUND_ALT = "Изобретение упоминает ещё одно общее свойство вечерней привычки."
_SYNTHETIC_PRACTICAL = "Изобретение предлагает спокойный вечерний распорядок как ближайший шаг."
_SYNTHETIC_PASSAGE_ID = "synthetic#exp0000"
_SYNTHETIC_SECOND_PASSAGE_ID = "synthetic#exp0001"


def _pack_entry(
    passage_id: str = _SYNTHETIC_PASSAGE_ID, text: str = "Синтетический фрагмент книги."
) -> dict[str, Any]:
    import hashlib

    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "synthetic-source",
        "section_id": f"synthetic-section-{passage_id[-1]}",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


def _unit_from_prompt(prompt: str) -> str:
    import re

    match = re.search(r"<response_unit>(.*?)</response_unit>", prompt, re.S)
    return match.group(1).strip() if match else prompt


class _SequencedRecordingAnswer:
    """Answer double returning scripted drafts and recording turn payloads."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.prompts: list[str] = []
        self.calls = 0

    async def ainvoke(self, messages: Any) -> Any:
        self.prompts.append(str(messages[-1].content))
        index = min(self.calls, len(self._texts) - 1)
        self.calls += 1
        return AIMessage(content=self._texts[index])


class _PracticalAwareVerifier:
    """Batched verifier double: practical text addresses intent, background does not."""

    supports_batched_verifier = True

    def __init__(
        self,
        *,
        practical_passage_id: str = _SYNTHETIC_PASSAGE_ID,
        background_passage_id: str = _SYNTHETIC_PASSAGE_ID,
    ) -> None:
        self._practical_passage_id = practical_passage_id
        self._background_passage_id = background_passage_id
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        self.calls += 1
        unit = _unit_from_prompt(prompt)
        practical = "ближайший шаг" in unit
        return {
            "verdicts": [
                {
                    "requires_book_evidence": True,
                    "supported": True,
                    "evidence_passage_ids": [
                        self._practical_passage_id if practical else self._background_passage_id
                    ],
                    "addresses_intent": practical,
                }
            ]
        }


class _RepairPlanner:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return {
            "mode": "retrieval",
            "resolved_intent": _SYNTHETIC_RESOLVED_INTENT,
            "queries": ["синтетический запрос о вечерней привычке"],
        }


class _Plan:
    def __init__(self) -> None:
        self.queries = ["синтетический запрос о вечерней привычке"]


@pytest.fixture(autouse=True)
def _enable_turn_time_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "true")


def test_relevance_only_failure_is_derived_from_verifier_fields_only() -> None:
    from aa.conversation.turn_pipeline import is_relevance_only_failure
    from aa.conversation.verifier_schema import GroundingResult, UnitVerdict

    def result(
        *,
        supported: bool = True,
        scope: str = "book",
        addresses_intent: bool = False,
        relevant: bool = False,
        unavailable: list[str] | None = None,
    ) -> GroundingResult:
        return GroundingResult(
            units=[
                UnitVerdict(
                    unit_id="u1",
                    scope=scope,  # type: ignore[arg-type]
                    supported=supported,
                    evidence_passage_ids=[_SYNTHETIC_PASSAGE_ID] if supported else [],
                    addresses_intent=addresses_intent,
                )
            ],
            all_required_supported=supported,
            unavailable_unit_ids=list(unavailable or []),
            answer_relevant=relevant,
            relevance_category="" if relevant else "irrelevant-citation",
        )

    assert is_relevance_only_failure(result(supported=True, addresses_intent=False, relevant=False))
    # Already relevant: nothing to repair.
    assert not is_relevance_only_failure(
        result(supported=True, addresses_intent=True, relevant=True)
    )
    # Unsupported claims need evidence, not a relevance instruction.
    assert not is_relevance_only_failure(
        result(supported=False, addresses_intent=False, relevant=False)
    )
    # An unavailable unit is a verifier outage, not a relevance verdict.
    assert not is_relevance_only_failure(
        result(supported=True, addresses_intent=False, relevant=False, unavailable=["u1"])
    )
    # A non-book unit makes relevance vacuous by construction.
    assert not is_relevance_only_failure(
        result(scope="conversation_glue", supported=True, addresses_intent=True, relevant=True)
    )
    assert not is_relevance_only_failure(None)


def test_relevance_repair_prompt_keeps_the_live_request_last() -> None:
    from aa.conversation.turn_pipeline import (
        RELEVANCE_REPAIR_INSTRUCTION,
        anchored_relevance_repair_prompt,
    )

    prompt = anchored_relevance_repair_prompt(_SYNTHETIC_USER_MESSAGE)

    assert prompt.endswith(_SYNTHETIC_USER_MESSAGE)
    instruction = prompt[: -len(_SYNTHETIC_USER_MESSAGE)]
    assert instruction.strip() == RELEVANCE_REPAIR_INSTRUCTION
    folded = instruction.casefold()
    assert "практич" in folded
    assert "ближайший шаг" in folded
    assert "общие сведения" in folded


def test_repair_focus_switches_to_relevance_only_for_that_failure() -> None:
    from aa.conversation.turn_pipeline import (
        anchored_relevance_repair_prompt,
        repair_focus_for_result,
    )
    from aa.conversation.verifier_schema import GroundingResult, UnitVerdict

    background = GroundingResult(
        units=[
            UnitVerdict(
                unit_id="u1",
                scope="book",
                supported=True,
                evidence_passage_ids=[_SYNTHETIC_PASSAGE_ID],
                addresses_intent=False,
            )
        ],
        all_required_supported=True,
        answer_relevant=False,
        relevance_category="irrelevant-citation",
    )
    unsupported = GroundingResult(
        units=[
            UnitVerdict(
                unit_id="u1",
                scope="book",
                supported=False,
                evidence_passage_ids=[],
                addresses_intent=False,
            )
        ],
        all_required_supported=False,
        answer_relevant=False,
        relevance_category="irrelevant-citation",
    )

    assert repair_focus_for_result(
        _SYNTHETIC_USER_MESSAGE, background, ["Синтетическая недостающая поддержка"]
    ) == anchored_relevance_repair_prompt(_SYNTHETIC_USER_MESSAGE)
    # Unsupported claims keep the missing-support supplement.
    focus = repair_focus_for_result(
        _SYNTHETIC_USER_MESSAGE, unsupported, ["Синтетическая недостающая поддержка"]
    )
    assert "Синтетическая недостающая поддержка" in focus
    assert focus.endswith(_SYNTHETIC_USER_MESSAGE)
    assert focus != anchored_relevance_repair_prompt(_SYNTHETIC_USER_MESSAGE)


async def test_relevance_only_repair_uses_same_pack_without_retrieval() -> None:
    from aa.conversation.turn_pipeline import (
        RELEVANCE_REPAIR_INSTRUCTION,
        run_v2_answer_turn,
    )

    answer = _SequencedRecordingAnswer([_SYNTHETIC_BACKGROUND, _SYNTHETIC_PRACTICAL])
    verifier = _PracticalAwareVerifier()

    outcome = await run_v2_answer_turn(
        user_message=_SYNTHETIC_USER_MESSAGE,
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=answer,
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent=_SYNTHETIC_RESOLVED_INTENT,
    )

    assert outcome["text"].strip() == _SYNTHETIC_PRACTICAL
    assert answer.calls == 2
    # The initial draft carried the raw request; the composition repair
    # carried the practical-guidance instruction and the request last.
    assert RELEVANCE_REPAIR_INSTRUCTION not in answer.prompts[0]
    assert RELEVANCE_REPAIR_INSTRUCTION in answer.prompts[1]
    assert _SYNTHETIC_USER_MESSAGE in answer.prompts[1]
    # The relevance verdict is still the verifier's: the background draft
    # failed and the practical draft passed on re-verification. A
    # relevance-only failure needs no planner or retrieval round.
    assert verifier.calls == 2
    telemetry = dict(outcome["telemetry"])
    assert telemetry["repair_rounds"] == 1
    assert telemetry["verifier_outcome"] == "passed-after-repair-existing-pack"


@pytest.mark.parametrize(
    ("retrieval_mode", "practical_passage_id", "expected_outcome"),
    [
        ("new-evidence", _SYNTHETIC_SECOND_PASSAGE_ID, "passed-after-repair"),
        (
            "duplicate-evidence",
            _SYNTHETIC_PASSAGE_ID,
            "passed-after-repair-existing-pack",
        ),
    ],
)
async def test_failed_composition_repair_keeps_relevance_focus_in_full_repair(
    monkeypatch: pytest.MonkeyPatch,
    retrieval_mode: str,
    practical_passage_id: str,
    expected_outcome: str,
) -> None:
    import aa.conversation.planner_node as planner_node
    import aa.conversation.retrieval_node as retrieval_node
    from aa.conversation.turn_pipeline import (
        RELEVANCE_REPAIR_INSTRUCTION,
        run_v2_answer_turn,
    )

    planner_calls = 0

    async def _fake_planner(*args: Any, **kwargs: Any) -> _Plan:
        nonlocal planner_calls
        _ = (args, kwargs)
        planner_calls += 1
        return _Plan()

    def _fake_pack_to_state(pack: Any) -> tuple[list[Any], list[dict[str, Any]]]:
        _ = pack
        if retrieval_mode == "new-evidence":
            return [], [
                dict(
                    _pack_entry(
                        _SYNTHETIC_SECOND_PASSAGE_ID,
                        "Синтетический дополнительный фрагмент книги.",
                    )
                )
            ]
        return [], [dict(_pack_entry())]

    async def _fake_retrieve(*args: Any, **kwargs: Any) -> object:
        _ = (args, kwargs)
        return object()

    monkeypatch.setattr(planner_node, "run_planner", _fake_planner)
    monkeypatch.setattr(retrieval_node, "aretrieve_with_semantic_selection", _fake_retrieve)
    monkeypatch.setattr(retrieval_node, "pack_to_state", _fake_pack_to_state)

    # The composition repair still answers only the background, so the
    # bounded full repair round runs; its regeneration must keep the
    # relevance focus rather than fall back to the raw request.
    answer = _SequencedRecordingAnswer(
        [_SYNTHETIC_BACKGROUND, _SYNTHETIC_BACKGROUND_ALT, _SYNTHETIC_PRACTICAL]
    )
    verifier = _PracticalAwareVerifier(practical_passage_id=practical_passage_id)

    outcome = await run_v2_answer_turn(
        user_message=_SYNTHETIC_USER_MESSAGE,
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=answer,
        verifier_model=verifier,
        planner_model=_RepairPlanner(),
        retrieval_index=object(),
        initial_query_count=12,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent=_SYNTHETIC_RESOLVED_INTENT,
    )

    assert outcome["text"].strip() == _SYNTHETIC_PRACTICAL
    assert planner_calls == 1
    assert answer.calls == 3
    assert RELEVANCE_REPAIR_INSTRUCTION not in answer.prompts[0]
    assert RELEVANCE_REPAIR_INSTRUCTION in answer.prompts[1]
    assert RELEVANCE_REPAIR_INSTRUCTION in answer.prompts[2]
    assert verifier.calls == 3
    telemetry = dict(outcome["telemetry"])
    assert telemetry["repair_rounds"] == 1
    assert telemetry["verifier_outcome"] == expected_outcome
