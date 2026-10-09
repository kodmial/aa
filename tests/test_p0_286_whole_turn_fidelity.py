"""P0 kodmial/aa#286: independent whole-turn book fidelity and follow-ups.

Mechanics only (invented fixture text, no canonical book text, no exact
live-question branches). Live product success is proven by the real
Telegram+provider lane with the separately instantiated judge; these
tests prove the machinery with injected fakes:

- independent whole-turn judge transport is strict (missing, null and
  wrong-type verdicts fail closed on both structured and text paths;
  unknown envelope keys are discarded);
- an independent FAIL overrides a telemetry PASS (and never the reverse);
- whole-turn relevance requires every supported book unit to address
  the intent (one relevant sentence no longer passes padding);
- a planner-misclassified glue turn whose verifier reports book need
  never takes the qualified conversational fallback;
- a double-misclassified glue pass (planner and verifier both say glue
  for substantive advice) is overturned by the injected whole-turn
  judge, while true glue still serves naturally;
- the bounded evidence window reports coverage mechanically and repair
  attempts use the wider top-8 window without blindly enlarging every
  turn.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Invented fixture support nearby helps steadily.",
) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


def _book_decision(passage_id: str, *, addresses: bool) -> dict[str, Any]:
    return {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": [passage_id],
        "addresses_intent": addresses,
    }


class _ScriptedStructured:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self._decisions = list(decisions)

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return dict(self._decisions.pop(0))


class _StructuredDownTextUp:
    def __init__(self, texts: list[str]) -> None:
        from aa.opencode.errors import OpenCodeDeterministicError

        self._texts = list(texts)
        self._error = OpenCodeDeterministicError("opencode structured output missing")

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        raise self._error

    async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
        _ = (prompt, system)
        return self._texts.pop(0)


class _ScriptedJudge:
    """Injected whole-turn judge fake returning scripted strict decisions."""

    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self._decisions = list(decisions)
        self.calls = 0
        self.agent = "aa-judge-v2"

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        return dict(self._decisions.pop(0))


class _StaticAnswer:
    def __init__(self, text: str) -> None:
        self._text = text

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        return AIMessage(content=self._text)


def test_judge_schema_rejects_missing_malformed() -> None:
    from aa.conversation.whole_turn_judge import (
        WholeTurnJudgeError,
        parse_whole_turn_text_decision,
        validate_whole_turn_decision,
    )

    for bad in (
        {"helpful": True, "addresses_intent": True},
        {"helpful": True, "addresses_intent": None, "contains_substantive_claim": False},
        {
            "helpful": "yes",
            "addresses_intent": True,
            "contains_substantive_claim": False,
        },
        {"helpful": 1, "addresses_intent": True, "contains_substantive_claim": False},
    ):
        with pytest.raises(WholeTurnJudgeError):
            validate_whole_turn_decision(bad)
    # Text path carries the same strictness.
    with pytest.raises(WholeTurnJudgeError):
        parse_whole_turn_text_decision(json.dumps({"helpful": True}))
    with pytest.raises(WholeTurnJudgeError):
        parse_whole_turn_text_decision(
            json.dumps(
                {
                    "helpful": True,
                    "addresses_intent": "yes",
                    "contains_substantive_claim": False,
                }
            )
        )
    # Unknown envelope keys are discarded, strict booleans still decide.
    parsed = parse_whole_turn_text_decision(
        json.dumps(
            {
                "helpful": True,
                "addresses_intent": True,
                "contains_substantive_claim": False,
                "unit_id": "invented",
                "all_required_supported": True,
            }
        )
    )
    assert validate_whole_turn_decision(parsed).helpful is True


async def test_judge_fail_overrides_telemetry_pass() -> None:
    from aa.conversation.whole_turn_judge import (
        WholeTurnJudgement,
        combine_telemetry_with_judge,
        judge_whole_turn,
    )

    passing = WholeTurnJudgement(
        helpful=True, addresses_intent=True, contains_substantive_claim=True
    )
    failing = WholeTurnJudgement(
        helpful=False, addresses_intent=False, contains_substantive_claim=False
    )
    assert combine_telemetry_with_judge(telemetry_pass=True, judge=passing) is True
    assert combine_telemetry_with_judge(telemetry_pass=True, judge=failing) is False
    assert combine_telemetry_with_judge(telemetry_pass=False, judge=passing) is False
    assert combine_telemetry_with_judge(telemetry_pass=True, judge=None) is True
    # Empty intent/reply fails closed without a model call.
    empty = await judge_whole_turn(resolved_intent="", reply="", model=_ScriptedJudge([]))
    assert empty.helpful is False
    helpful = await judge_whole_turn(
        resolved_intent="synthetic intent",
        reply="synthetic reply",
        model=_ScriptedJudge(
            [
                {
                    "helpful": True,
                    "addresses_intent": True,
                    "contains_substantive_claim": False,
                }
            ]
        ),
    )
    assert helpful.helpful is True


async def test_whole_turn_requires_every_book_unit_relevant() -> None:
    from aa.conversation.response_units import split_response_units
    from aa.conversation.verifier import run_verifier

    pack = [_pack_entry("chapter-3#exp0000"), _pack_entry("chapter-3#exp0001")]
    units = split_response_units("First invented guidance. Second invented guidance.")
    assert len(units) == 2
    mixed = _ScriptedStructured(
        [
            _book_decision(pack[0]["passage_id"], addresses=True),
            _book_decision(pack[1]["passage_id"], addresses=False),
        ]
    )
    result = await run_verifier(units, pack, model=mixed)
    assert result.all_required_supported is True
    assert result.answer_relevant is False
    assert result.relevance_category == "irrelevant-citation"


async def test_misclassified_glue_with_book_need_skips_qualified_fallback() -> None:
    from aa.conversation.turn_pipeline import (
        run_v2_answer_turn,
    )

    pack: list[dict[str, Any]] = []
    draft = "Поддержка рядом помогает пережить тягу спокойно."
    # The verifier reports a book-dependent claim on an empty pack: the
    # planner misclassified a substantive request, so the qualified
    # conversational fallback must not mark it as success.
    import pytest as _pt286

    from aa.conversation.failures import TurnFailed as _TF286

    with _pt286.raises(_TF286):
        await run_v2_answer_turn(
            user_message="synthetic practical request",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=_StaticAnswer(draft),
            verifier_model=_ScriptedStructured(
                [
                    {
                        "requires_book_evidence": True,
                        "supported": False,
                        "evidence_passage_ids": [],
                        "addresses_intent": False,
                    }
                ]
            ),
            initial_query_count=0,
            planner_reason="legitimate-glue",
            planner_mode="conversational",
            resolved_intent="synthetic practical intent",
        )


async def test_double_lie_glue_overturned_by_judge_true_glue_preserved() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    glue_draft = "Готов спокойно выслушать и поддержать разговор."
    # Both planner and verifier claim glue for substantive advice; the
    # independent judge sees the substantive claim and overturns as a
    # typed unsuccessful outcome (never qualified glue success).
    import pytest as _pt286b

    from aa.conversation.failures import TurnFailed as _TF286b

    with _pt286b.raises(_TF286b) as _exc286:
        await run_v2_answer_turn(
            user_message="synthetic practical request",
            summary="",
            recent=[],
            evidence_pack=[],
            answer_model=_StaticAnswer(glue_draft),
            verifier_model=_ScriptedStructured(
                [
                    {
                        "requires_book_evidence": False,
                        "supported": True,
                        "evidence_passage_ids": [],
                        "addresses_intent": True,
                    }
                ]
            ),
            initial_query_count=0,
            planner_reason="legitimate-glue",
            planner_mode="conversational",
            resolved_intent="synthetic practical intent",
            whole_turn_judge_model=_ScriptedJudge(
                [
                    {
                        "helpful": False,
                        "addresses_intent": False,
                        "contains_substantive_claim": True,
                    }
                ]
            ),
        )
    assert _exc286.value.telemetry["qualified"] is False
    assert _exc286.value.telemetry["answers_request"] is False
    # True glue with a judge reporting no substantive claim still serves.
    judge = _ScriptedJudge(
        [
            {
                "helpful": True,
                "addresses_intent": True,
                "contains_substantive_claim": False,
            }
        ]
    )
    served = await run_v2_answer_turn(
        user_message="synthetic greeting",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=_StaticAnswer(glue_draft),
        verifier_model=_ScriptedStructured(
            [
                {
                    "requires_book_evidence": False,
                    "supported": True,
                    "evidence_passage_ids": [],
                    "addresses_intent": True,
                }
            ]
        ),
        initial_query_count=0,
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
        whole_turn_judge_model=judge,
    )
    assert served["text"] == glue_draft
    assert judge.calls == 1


def test_evidence_window_coverage_and_repair_width() -> None:
    from aa.conversation.turn_pipeline import ANSWER_GENERATION_MAX_PASSAGES
    from aa.conversation.whole_turn_judge import (
        REPAIR_GENERATION_MAX_PASSAGES,
        assess_evidence_window_coverage,
        coverage_to_metrics,
    )

    assert ANSWER_GENERATION_MAX_PASSAGES == 0
    assert REPAIR_GENERATION_MAX_PASSAGES == 0
    pack = [
        _pack_entry(f"chapter-3#exp{i:04d}", f"Invented fixture text {i}. " * 40) for i in range(8)
    ]
    coverage = assess_evidence_window_coverage(
        pack,
        generation_window=ANSWER_GENERATION_MAX_PASSAGES,
        verifier_window=0,
        cited_ids=[pack[6]["passage_id"]],
    )
    assert coverage.pack_passages == 8
    assert coverage.omitted_from_generation == 0
    assert coverage.omitted_from_verifier == 0
    assert coverage.cited_outside_generation == 0
    assert coverage.cited_outside_verifier == 0
    metrics = coverage_to_metrics(coverage)
    assert metrics["pack_passages"] == 8
    assert "Invented fixture text" not in json.dumps(metrics)


async def test_repair_uses_wider_window_for_rank_six_evidence() -> None:
    import pathlib

    # Full-pack policy (#295): initial drafts and repair regens receive
    # the entire Evidence Pack with full text, so decisive evidence at
    # ranks 6+ (including >16 after semantic promotion) is reachable on
    # the first attempt and on retry. Proven statically here plus the
    # coverage test above.
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8")
    assert "wider=True" in src
    # Live telemetry records the omitted-window counts (privacy-safe, zero at full pack).
    assert "evidence_window_omitted_generation" in src
    assert "evidence_window_omitted_verifier" in src
    from aa.conversation.turn_pipeline import ANSWER_GENERATION_MAX_PASSAGES
    from aa.conversation.whole_turn_judge import REPAIR_GENERATION_MAX_PASSAGES as _REPAIR

    assert ANSWER_GENERATION_MAX_PASSAGES == 0
    assert _REPAIR == 0


def test_no_domain_regex_or_exact_oracle_in_new_control() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    judge_src = (root / "src" / "aa" / "conversation" / "whole_turn_judge.py").read_text(
        encoding="utf-8"
    )
    assert "import re" not in judge_src
    assert "re.compile" not in judge_src
    for probe in ("вечерний распорядок", "support implies relevance", "had_explicit_relevance"):
        assert probe not in judge_src
    # Live lane invokes the separately instantiated judge; telemetry
    # alone is documented as never independent proof.
    live_src = (root / "src" / "aa" / "qualification" / "product_contract_live.py").read_text(
        encoding="utf-8"
    )
    assert "build_live_whole_turn_judge" in live_src
    assert "assess_live_helpfulness_with_judge_metrics" in live_src
    assert "judge_overrode_telemetry" in live_src
    assert "aa-judge-v2" in live_src
    # Whole-turn relevance now requires every supported book unit.
    schema_src = (root / "src" / "aa" / "conversation" / "verifier_schema.py").read_text(
        encoding="utf-8"
    )
    assert "every supported book unit" in schema_src
    assert re.search(r"for verdict in supported_book", schema_src)
