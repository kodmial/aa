"""P0 regression tests for issue #145: fallback, latency and READY state.

Proves through the exact production boundary that ordinary turns carry
privacy-safe stage telemetry, glue-judged turns skip the expensive repair
loop, the runtime publishes STARTING/READY/STOPPED, and #7 cannot PASS
offline without real live Telegram evidence.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage

from aa.app import Application
from aa.config import Settings
from aa.control.campaign import runtime_phase
from aa.conversation.turn_telemetry import (
    TurnTelemetry,
    percentile,
    summarize_latencies,
)
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime


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


class _AnswerModel:
    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        return AIMessage(content=self._drafts.pop(0))


class _VerifierModel:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = list(results)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        return dict(self._results.pop(0))


class _PlannerModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        raise AssertionError("repair planner must not run for glue-judged turns")


def test_telemetry_carries_only_counts_and_latencies() -> None:
    telemetry = TurnTelemetry(
        planner_outcome="ok",
        planner_latency_ms=12.5,
        planner_query_count=12,
        retrieval_outcome="evidence-ready",
        retrieval_latency_ms=44.0,
        total_latency_ms=120.0,
        reply_len=42,
    )
    payload = telemetry.to_safe_dict()
    assert payload["planner_query_count"] == 12
    assert payload["total_latency_ms"] == 120.0
    text = str(payload).casefold()
    assert "тяга" not in text
    for forbidden in ("utterance", "answer", "evidence_text", "transcript"):
        assert forbidden not in payload


def test_percentile_and_summary_helpers() -> None:
    assert percentile([], 50) == 0.0
    assert summarize_latencies([])["count"] == 0.0
    summary = summarize_latencies([0.1, 0.2, 0.3, 0.4])
    assert summary["count"] == 4.0
    assert summary["p50_s"] >= 0.1
    assert summary["p95_s"] >= summary["p50_s"]


async def test_glue_turn_skips_repair_and_reports_telemetry() -> None:
    from aa.conversation.response_units import split_response_units
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    draft = "Понимаю. Расскажите, что сейчас важнее всего?"
    units = split_response_units(draft)
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [],
                    }
                    for unit in units
                ],
                "all_required_supported": False,
            }
        ]
    )
    planner = _PlannerModel()
    outcome = await run_v2_answer_turn(
        user_message="А что вы можете?",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=answer,
        verifier_model=verifier,
        planner_model=planner,
        retrieval_index=object(),
    )
    assert outcome["rounds"] == 0
    assert planner.calls == 0
    assert answer.calls == 1
    assert verifier.calls == 1
    telemetry = outcome["telemetry"]
    assert telemetry["initial_pack_empty"] is True
    assert telemetry["planner_outcome"] == "skipped-glue"
    assert telemetry["retrieval_outcome"] == "skipped-glue"
    assert telemetry["total_latency_ms"] >= 0.0


async def test_substantive_turn_still_repairs_with_evidence() -> None:
    from aa.conversation.response_units import split_response_units
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    first_pack = [_pack_entry()]
    first_draft = "Поддержка рядом помогает. Тяга лечится луной за вечер."
    repaired_draft = "Поддержка рядом помогает пережить тягу спокойно."
    repaired_units = split_response_units(repaired_draft)
    answer = _AnswerModel([first_draft, repaired_draft])

    class _TwelvePlanner:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            return {"queries": [f"запрос про поддержку {idx}" for idx in range(12)]}

    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [first_pack[0]["passage_id"]],
                    },
                    {
                        "unit_id": "u2",
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [first_pack[0]["passage_id"]],
                    },
                ],
                "all_required_supported": False,
            },
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": ["chapter-3#exp0001"],
                    }
                    for unit in repaired_units
                ],
                "all_required_supported": True,
            },
        ]
    )
    planner = _TwelvePlanner()

    import hashlib as _hashlib

    from aa.retrieval import evidence as evidence_mod

    def _fake_retrieve(index: Any, queries: object, *, config: Any = None) -> Any:
        from aa.retrieval.evidence import EvidencePack, EvidencePassageData

        text = "Фиктивная поддержка рядом помогает пережить тягу спокойно."
        passage = EvidencePassageData(
            passage_id="chapter-3#exp0001",
            exact_text=text,
            source_id="ru-fourth-edition-txt",
            section_id="chapter-3",
            child_chunk_ids=("chapter-3:ru:1",),
            char_start=120,
            char_end=240,
            text_sha256=_hashlib.sha256(text.encode()).hexdigest(),
            source_sha256="s" * 64,
        )
        return EvidencePack(
            passages=(passage,), total_tokens=10, corpus_version="v", retrieval_metadata={}
        )

    import pytest

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(evidence_mod, "retrieve_evidence", _fake_retrieve)
    try:
        outcome = await run_v2_answer_turn(
            user_message="что помогает при тяге?",
            summary="",
            recent=[],
            evidence_pack=first_pack,
            answer_model=answer,
            verifier_model=verifier,
            planner_model=planner,
            retrieval_index=object(),
        )
    finally:
        monkeypatch.undo()
    assert outcome["rounds"] == 1
    assert outcome["text"] == repaired_draft
    assert outcome["telemetry"]["repair_rounds"] == 1


async def test_runtime_publishes_starting_ready_stopped() -> None:
    from aa.app import RUNTIME_READY, RUNTIME_STARTING, RUNTIME_STOPPED

    settings = Settings.from_env({})
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )
    app = Application(settings, opencode_runtime=runtime)
    assert app.runtime_state == RUNTIME_STOPPED
    await app.start()
    try:
        assert app.running
        assert app.runtime_state == RUNTIME_READY
        assert app.runtime_state != RUNTIME_STARTING
    finally:
        await app.stop()
    assert app.runtime_state == RUNTIME_STOPPED


def test_bot_status_distinguishes_starting_vs_ready() -> None:
    assert runtime_phase(pending_dispatch=True, runtime_active=False, stopped=False) == "STARTING"
    assert (
        runtime_phase(pending_dispatch=False, runtime_active=True, stopped=False, has_ready=True)
        == "READY"
    )
    assert (
        runtime_phase(pending_dispatch=False, runtime_active=True, stopped=False, has_ready=False)
        == "STARTING"
    )
    assert runtime_phase(pending_dispatch=False, runtime_active=False, stopped=False) == "STOPPED"
    assert runtime_phase(pending_dispatch=False, runtime_active=True, stopped=True) == "STOPPED"


async def test_live_evidence_lane_is_incomplete_offline(monkeypatch: Any) -> None:
    from aa.qualification.product_contract_live import run_live_telegram_evidence_lane

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_BOOK_AGE_IDENTITY", raising=False)
    result = await run_live_telegram_evidence_lane()
    assert result.lane == "live-telegram-evidence"
    assert result.status == "INCOMPLETE"
    assert result.incomplete
    assert result.metrics["latency_budget_s"] == 30.0
    assert "latency_p50_s" in result.metrics
    assert "latency_p95_s" in result.metrics


def test_offline_aggregate_cannot_pass_without_live_evidence() -> None:
    from aa.qualification.product_contract_live import decide_status

    assert decide_status(["PASS", "PASS", "PASS", "PASS", "INCOMPLETE"]) == "INCOMPLETE"
    assert decide_status(["PASS", "PASS", "PASS", "PASS", "FAIL"]) == "FAIL"


def test_answer_and_verifier_prompts_scope_capability_without_whitelist() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    agent = (root / "prompts" / "aa-agent-system-v2.md").read_text(encoding="utf-8")
    verifier = (root / "prompts" / "aa-verifier-system-v2.md").read_text(encoding="utf-8")
    assert "all substantive ideas" in agent
    assert "Return only the user-facing Russian reply." in agent
    assert "no passages" in agent
    assert "general offer" in agent.casefold() or "general capability" in agent.casefold()
    assert "general offer" in verifier.casefold()
    # No exact-question whitelist: observed live inputs never appear verbatim.
    for prompt_text in (agent, verifier):
        assert "37422302821" not in prompt_text
