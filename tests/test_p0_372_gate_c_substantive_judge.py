"""P0 kodmial/aa#372 SYSTEMIC: substantive delivery must enforce the independent judge.

Systemic failure class (not one prompt): the verifier's per-unit
``addresses_intent`` and the qualification whole-turn judge's ``helpful``
are two independent model judgments that diverge stochastically.
Successive narrow repairs aligned prompt wording for one scenario
family at a time (background restatement, greeting glue, irrelevant
citation), and each time the next paraphrase passed the internal
adequacy gate yet failed the external judge -- the production path
never ran that judge internally. The production graph did not even
construct the judge model, so the glue-only judge guards in the answer
pipeline were dead code on the live path while Gate C judged every
substantive turn with it.

Architecture repair (generic, no exact-question branches, no keyword
tables, Product Contract #110 unchanged): the production graph binds
the same separately-identified judge model, and the answer pipeline
gates every substantive qualified delivery on its explicit verdict. An
explicit negative withholds the delivery in favor of the existing
bounded regen/narrow/fail path; a missing judge, exhausted call bound,
missing budget slice, timeout, outage, or unavailable verdict
preserves the historical adequacy path. Verdicts for byte-identical
text are cached per turn by digest so the common serve costs one judge
round.

All strings below are synthetic fixtures for this failure class, never
the literal qualification prompts.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

# Synthetic fixtures for the failure class (never qualification prompts).
_SYNTHETIC_USER_MESSAGE = "Синтетический вопрос о вечернем напряжении и спокойном распорядке"
_SYNTHETIC_RESOLVED_INTENT = "синтетическое намерение разобраться с вечерним напряжением"
_SYNTHETIC_DRAFT = (
    "Понимаю, вечернее напряжение даётся непросто. "
    "Спокойный распорядок и поддержка рядом помогают пройти этот час."
)
_SYNTHETIC_PACK_TEXT = "Синтетический отрывок о спокойном вечернем распорядке."
_SYNTHETIC_PACK_ID = "chapter-3#exp0000"


def _pack_entry() -> dict[str, Any]:
    text = _SYNTHETIC_PACK_TEXT
    return {
        "passage_id": _SYNTHETIC_PACK_ID,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


class _StubJudge:
    """Separately-identified whole-turn judge double with a fixed verdict."""

    def __init__(self, *, helpful: bool) -> None:
        self._helpful = helpful
        self.calls = 0
        self.agent = "stub-judge-372"

    async def ainvoke_structured(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, object],
        retry_count: int = 1,
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        return {
            "helpful": self._helpful,
            "addresses_intent": self._helpful,
            "contains_substantive_claim": True,
        }


def _install_pipeline_doubles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub draft generation and per-unit verification inside the pipeline."""
    import aa.conversation.turn_pipeline as pipeline
    from aa.conversation.verifier_schema import (
        GroundingResult,
        UnitVerdict,
        validate_grounding_result,
    )

    async def _fake_generate_draft(*args: Any, **kwargs: Any) -> str:
        _ = (args, kwargs)
        return _SYNTHETIC_DRAFT

    async def _fake_run_verifier(units: Any, passages: Any, **kwargs: Any) -> GroundingResult:
        _ = (passages, kwargs)
        verdicts = [
            UnitVerdict(
                unit_id=unit.unit_id,
                scope="book",
                supported=True,
                evidence_passage_ids=[_SYNTHETIC_PACK_ID],
                addresses_intent=True,
                origin="book_claim",
            )
            for unit in units
        ]
        return validate_grounding_result(
            GroundingResult(
                verified=True,
                units=verdicts,
                all_required_supported=True,
                unavailable_unit_ids=[],
            ),
            expected_unit_ids=[unit.unit_id for unit in units],
        )

    monkeypatch.setattr(pipeline, "generate_draft", _fake_generate_draft)
    monkeypatch.setattr(pipeline, "run_verifier", _fake_run_verifier)


def _substantive_kwargs(judge: Any) -> dict[str, Any]:
    return {
        "user_message": _SYNTHETIC_USER_MESSAGE,
        "summary": "",
        "recent": [],
        "evidence_pack": [_pack_entry()],
        "answer_model": object(),
        "verifier_model": object(),
        "planner_model": None,
        "retrieval_index": None,
        "initial_query_count": 3,
        "planner_reason": "substantive-with-queries",
        "planner_mode": "retrieval",
        "resolved_intent": _SYNTHETIC_RESOLVED_INTENT,
        "whole_turn_judge_model": judge,
    }


async def test_judge_reject_withholds_qualified_substantive_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verifier-passing draft the judge finds unhelpful is never served qualified."""
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_pipeline_doubles(monkeypatch)
    judge = _StubJudge(helpful=False)
    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(**_substantive_kwargs(judge))
    # One judge round decides; byte-identical regen/narrow candidates reuse it.
    assert judge.calls == 1


async def test_judge_approve_serves_qualified_substantive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verifier-passing draft the judge finds helpful is served qualified."""
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_pipeline_doubles(monkeypatch)
    judge = _StubJudge(helpful=True)
    outcome = await run_v2_answer_turn(**_substantive_kwargs(judge))
    assert outcome["text"] == _SYNTHETIC_DRAFT
    telemetry = dict(outcome["telemetry"])
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["qualified"] is True
    assert telemetry["independent_judge_helpful"] is True
    assert judge.calls == 1


async def test_no_judge_preserves_legacy_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a bound judge the historical adequacy-only path is unchanged."""
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_pipeline_doubles(monkeypatch)
    outcome = await run_v2_answer_turn(**_substantive_kwargs(None))
    assert outcome["text"] == _SYNTHETIC_DRAFT
    telemetry = dict(outcome["telemetry"])
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["qualified"] is True


def test_judge_model_is_independently_identified_and_pinned() -> None:
    """The production judge keeps its own audit identity with no fallback."""
    from aa.conversation.model_adapter import build_judge_model
    from aa.conversation.whole_turn_judge import WHOLE_TURN_JUDGE_AGENT_V2

    client: Any = object()
    model = build_judge_model(client, primary_model="")
    assert model.agent == WHOLE_TURN_JUDGE_AGENT_V2
    assert model.agent != "aa-v2"
    assert model.agent != "aa-verifier-v2"
    assert model.fallback_model == ""
    assert model.transport_agent == ""
    assert model.primary_model.strip() != ""


def test_production_graph_threads_judge_to_answer() -> None:
    """The compiled graph accepts and forwards the judge model to the answer node."""
    import inspect

    from aa.conversation import graph as graph_mod
    from aa.conversation import turn_pipeline as pipeline_mod

    assert "whole_turn_judge_model" in inspect.signature(graph_mod.build_turn_graph).parameters
    assert (
        "whole_turn_judge_model" in inspect.signature(pipeline_mod.answer_pipeline_node).parameters
    )
    assert "whole_turn_judge_model" in inspect.signature(pipeline_mod.run_v2_answer_turn).parameters


def test_no_exact_question_special_cases() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    product_sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph_runtime.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "model_adapter.py").read_text(encoding="utf-8"),
    ]
    for fragment in (
        "Чем ты вообще можешь быть полезен",
        "Слушай, а ты тут вообще чем помочь можешь",
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
        "Дома снова ссора",
        "Поругались дома из-за бухла",
        "покупать акции",
        "Какой телефон сейчас выгоднее купить",
        "Привет! К вечеру очень тянет выпить",
        "Привет, подскажи че делать когда под вечер тянеет выпить",
    ):
        for source in product_sources:
            assert fragment not in source
    # This test's own synthetic pack fixture must not leak into product sources.
    for source in product_sources:
        assert "Синтетический отрывок о спокойном вечернем распорядке" not in source
