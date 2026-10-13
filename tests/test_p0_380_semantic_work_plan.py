"""P0 kodmial/aa#380: single coherent budget-aware semantic work plan.

Proves the systemic runtime architecture replacement through invented
fixture text only (no canonical book text, no qualification literals,
no exact live-question special cases):

- privacy-safe stage critical-path DAG on the real graph path
  (per-turn hashed id, in-flight/done ops, provider
  attempts/timeouts/cancellation/429, token usage, retrieval
  selection/coverage iterations, verifier unit counts and repeated
  evidence sizes, whole-turn judge, final certificate);
- one batched single-round verifier call per draft with the Evidence
  Pack materialized once (1, 2 and 8+ response units), with strict
  per-unit completeness, citation, integrity, quote/provenance and
  intent-relevance gates (missing/duplicate/invalid verdicts,
  poisoned evidence and ambiguous batches fail closed; 429
  propagates);
- deadline-aware provider calls with reserved generator/verifier/
  judge/finalization slices and a genuinely enforced hard deadline
  (fake slow/blocked/cancellation-resistant providers, concurrency,
  bounded fanout, no orphan requests, no full pipeline repeat on
  no-progress);
- unchanged Gate E SLO constants and PASS semantics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aa.conversation.failures import TurnFailed
from aa.conversation.graph import gate_node, turn_input
from aa.conversation.graph_state import TurnState
from aa.conversation.response_units import split_response_units
from aa.conversation.semantic_work_plan import (
    ANSWER_MIN_SLICE_S,
    FINALIZE_RESERVE_S,
    GATE_E_P95_TARGET_S,
    HARD_TURN_SLO_S,
    INTERNAL_TURN_BUDGET_S,
    JUDGE_CALL_CAP_S,
    MAX_FULL_REPAIR_ROUNDS,
    VERIFIER_MIN_SLICE_S,
    VERIFIER_ROUND_MODE,
    RepairProgress,
    SemanticTurnDAG,
    clear_dag_registry,
    dag_for_state,
    ensure_turn_dag,
    estimate_tokens_for_chars,
    new_turn_dag_id,
    new_work_plan,
    repair_made_progress,
)
from aa.conversation.turn_pipeline import (
    MAX_TARGETED_REPAIR_ROUNDS,
    TURN_ANSWER_MIN_SLICE_S,
    TURN_END_TO_END_BUDGET_S,
    TURN_VERIFIER_MIN_SLICE_S,
    run_v2_answer_turn,
)
from aa.conversation.verifier import (
    batched_text_json_suffix,
    build_batched_verifier_text,
    clear_verifier_capability_cache,
    coerce_batched_verdicts,
    model_supports_batched_verifier,
    parse_batched_text_json_decision,
    run_batched_verifier,
    run_verifier,
)
from aa.conversation.verifier_schema import (
    VerifierValidationError,
    validate_batched_decisions,
    verifier_batched_json_schema,
    verifier_single_json_schema,
)
from aa.opencode.errors import (
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
    OpenCodeTimeoutError,
)


@pytest.fixture(autouse=True)
def _isolated_verifier_and_dag_state() -> Any:
    clear_verifier_capability_cache()
    clear_dag_registry()
    yield
    clear_verifier_capability_cache()
    clear_dag_registry()


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
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


def _decision(
    *,
    book: bool,
    supported: bool,
    cites: list[str],
    relevant: bool = True,
    origin: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requires_book_evidence": book,
        "supported": supported,
        "evidence_passage_ids": list(cites),
        "addresses_intent": relevant,
    }
    if origin is not None:
        payload["claim_origin"] = origin
    return payload


TEN_UNIT_DRAFT = (
    "Поддержка рядом помогает пережить тягу. "
    "Спокойный разговор снижает напряжение. "
    "Честный рассказ о трудностях облегчает вечер. "
    "Прогулка на свежем воздухе отвлекает от мыслей. "
    "Тёплый чай помогает расслабиться без алкоголя. "
    "Звонок другу возвращает чувство опоры. "
    "Короткая пауза перед решением сохраняет трезвость. "
    "Запись переживаний проясняет голову. "
    "Ранний сон восстанавливает силы. "
    "Утренний план делает день предсказуемым."
)

SINGLE_UNIT_DRAFT = "Поддержка рядом помогает пережить тягу сегодня."
TWO_UNIT_DRAFT = "Понимаю. Поддержка рядом помогает пережить тягу сегодня."


class _ScriptedAAVerifier:
    """Scripted verifier serving batched or single decisions on demand."""

    def __init__(
        self,
        payloads: list[Any],
        *,
        batched: bool,
        text_fallback: str | None = None,
        advertise: bool = True,
    ) -> None:
        self._payloads = list(payloads)
        self._batched = batched
        self._text_fallback = text_fallback
        self.calls = 0
        self.text_calls = 0
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []
        if batched and advertise:
            self.supports_batched_verifier = True

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (system, retry_count)
        self.calls += 1
        self.prompts.append(prompt)
        self.schemas.append(dict(schema))
        if not self._payloads:
            raise AssertionError("verifier called more times than scripted")
        payload = self._payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        if self._batched:
            decisions = [dict(item) for item in payload]
            return {"verdicts": decisions}
        return dict(payload)

    async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
        _ = (prompt, system)
        self.text_calls += 1
        if self._text_fallback is None:
            raise AssertionError("text path called without scripted reply")
        return self._text_fallback


class _AnswerModel:
    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        self.calls += 1
        if not self._drafts:
            raise AssertionError("answer model called more times than scripted")
        return AIMessage(content=self._drafts.pop(0))


class _PlannerModel:
    def __init__(self, plans: list[dict[str, Any]]) -> None:
        self._plans = list(plans)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        if not self._plans:
            raise AssertionError("planner called more times than scripted")
        return dict(self._plans.pop(0))


def _twelve_queries(base: str) -> list[str]:
    return [f"{base} вариант {index}" for index in range(12)]


# ---------------------------------------------------------------------------
# Part 1: privacy-safe critical-path DAG and coherent work plan.
# ---------------------------------------------------------------------------


def test_dag_ids_are_hashed_unique_and_text_free() -> None:
    first = new_turn_dag_id()
    second = new_turn_dag_id()
    assert re.fullmatch(r"[0-9a-f]{16}", first) is not None
    assert first != second


def test_dag_snapshot_never_carries_user_or_book_text() -> None:
    user_sentinel = "Секретный пользовательский текст 7ф3я"
    book_sentinel = "Канонический отрывок книги 9ц2ю"
    dag = SemanticTurnDAG()
    dag.start_op("planner")
    dag.provider_attempt("planner", input_tokens=123)
    dag.finish_op("planner", ok=True, category="planned")
    dag.record_selection_iteration(candidates=40, selected=3)
    dag.record_coverage_iteration(covered=2, uncovered=0)
    dag.record_verifier_round(unit_count=8, evidence_chars=4000, materialized_chars=4000)
    dag.record_judge_round()
    dag.finish_op("retrieval", ok=True, category="ready")
    dumped = json.dumps(dag.snapshot(), ensure_ascii=False)
    assert user_sentinel not in dumped
    assert book_sentinel not in dumped
    assert dag.in_flight_ops() == []
    # Only fixed safe keys travel outward.
    snapshot = dag.snapshot()
    assert set(snapshot) == {
        "dag_id",
        "elapsed_ms",
        "ops",
        "selection_iterations",
        "coverage_iterations",
        "coverage_expansions",
        "coverage_searches",
        "verifier_units",
        "verifier_evidence_chars",
        "verifier_materialized_chars",
        "judge_rounds",
        "in_flight",
    }


def test_dag_op_lifecycle_and_provider_counters() -> None:
    dag = SemanticTurnDAG()
    assert dag.in_flight_ops() == []
    dag.start_op("answer")
    assert dag.in_flight_ops() == ["answer"]
    dag.provider_attempt("answer", input_tokens=50)
    dag.provider_attempt("answer", input_tokens=10)
    dag.provider_timeout("answer")
    dag.provider_cancelled("answer")
    dag.provider_rate_limited("answer")
    dag.record_token_usage("answer", output_tokens=20)
    dag.finish_op("answer", ok=False, category="attempt-timeout")
    assert dag.in_flight_ops() == []
    ops = {item["op"]: item for item in dag.snapshot()["ops"]}
    record = ops["answer"]
    assert record["attempts"] == 2
    assert record["timeouts"] == 1
    assert record["cancellations"] == 1
    assert record["rate_limits"] == 1
    assert record["input_tokens"] == 60
    assert record["output_tokens"] == 20
    assert record["done"] is True


def test_dag_verifier_accounting_for_varying_unit_counts() -> None:
    dag = SemanticTurnDAG()
    dag.record_verifier_round(unit_count=1, evidence_chars=2000, materialized_chars=2000)
    dag.record_verifier_round(unit_count=2, evidence_chars=2000, materialized_chars=2000)
    dag.record_verifier_round(unit_count=10, evidence_chars=2000, materialized_chars=2000)
    snapshot = dag.snapshot()
    assert snapshot["verifier_units"] == 13
    assert snapshot["verifier_evidence_chars"] == 6000
    assert snapshot["verifier_materialized_chars"] == 6000


def test_state_shares_one_dag_across_nodes() -> None:
    state: TurnState = turn_input("Здравствуйте, нужна поддержка.")
    first = ensure_turn_dag(state)
    second = ensure_turn_dag(state)
    assert first is second
    assert dag_for_state(state) is first
    assert str(state.get("semantic_dag_id", "")) == first.dag_id


def test_work_plan_constants_match_authoritative_budgets() -> None:
    from aa.conversation.turn_budget import TURN_END_TO_END_BUDGET_S as _budget_s
    from aa.conversation.turn_budget import TURN_HARD_SLO_MS as _hard_ms
    from aa.conversation.turn_budget import TURN_P95_TARGET_MS as _p95_ms

    assert INTERNAL_TURN_BUDGET_S == TURN_END_TO_END_BUDGET_S == _budget_s == 105.0
    assert INTERNAL_TURN_BUDGET_S < HARD_TURN_SLO_S
    assert HARD_TURN_SLO_S * 1000.0 == _hard_ms == 120_000.0
    assert GATE_E_P95_TARGET_S * 1000.0 == _p95_ms == 60_000.0
    assert ANSWER_MIN_SLICE_S == TURN_ANSWER_MIN_SLICE_S == 1.0
    assert VERIFIER_MIN_SLICE_S == TURN_VERIFIER_MIN_SLICE_S == 3.0
    assert FINALIZE_RESERVE_S == 2.0
    assert JUDGE_CALL_CAP_S == 20.0
    assert VERIFIER_ROUND_MODE == "batched-single-round"
    assert MAX_FULL_REPAIR_ROUNDS == MAX_TARGETED_REPAIR_ROUNDS == 2


def test_work_plan_deadline_gate_before_provider_call() -> None:
    plan = new_work_plan(SemanticTurnDAG())
    assert plan.allows_provider_call(remaining_s=105.0, need_s=1.0) is True
    assert plan.allows_provider_call(remaining_s=0.5, need_s=1.0) is False
    assert plan.allows_provider_call(remaining_s=4.0, need_s=1.0, reserve_for="verifier") is True
    assert plan.allows_provider_call(remaining_s=3.5, need_s=1.0, reserve_for="verifier") is False
    assert plan.allows_provider_call(remaining_s=-1.0, need_s=1.0) is False
    assert plan.allows_provider_call(remaining_s=10.0, need_s=0.0) is False
    assert plan.verifier_round_mode == "batched-single-round"
    assert plan.repair_full_round_allowed(full_rounds_used=0) is True
    assert plan.repair_full_round_allowed(full_rounds_used=1) is True
    assert plan.repair_full_round_allowed(full_rounds_used=2) is False
    assert plan.describe()["max_full_repair_rounds"] == 2


def test_work_plan_judge_slice_matches_legacy_budget_neutral_math() -> None:
    plan = new_work_plan(SemanticTurnDAG())
    for remaining in (0.0, 1.0, 2.9, 3.0, 3.5, 8.0, 25.0, 105.0):
        expected_reserve = max(0.0, float(TURN_VERIFIER_MIN_SLICE_S))
        expected = min(max(0.0, remaining - expected_reserve), 20.0)
        assert plan.judge_slice(remaining) == pytest.approx(max(0.0, expected))


def test_repair_progress_gate_predicate() -> None:
    before = RepairProgress(pack_passages=2, supported_units=0)
    assert repair_made_progress(before, RepairProgress(3, 0)) is True
    assert repair_made_progress(before, RepairProgress(2, 1)) is True
    assert repair_made_progress(before, RepairProgress(2, 0)) is False
    assert repair_made_progress(before, RepairProgress(1, 0)) is False
    assert repair_made_progress(before, before) is False


def test_estimate_tokens_counts_without_text() -> None:
    assert estimate_tokens_for_chars(0) == 0
    assert estimate_tokens_for_chars(-5) == 0
    assert estimate_tokens_for_chars(4000) > 0


async def test_gate_node_creates_dag_without_text_leak() -> None:
    state: TurnState = turn_input("У меня начались судороги, помогите")
    update = await gate_node(state)
    assert update["route"] == "emergency"
    dag_id = str(update.get("semantic_dag_id", ""))
    assert re.fullmatch(r"[0-9a-f]{16}", dag_id) is not None
    dag = dag_for_state({**dict(state), "semantic_dag_id": dag_id})
    assert dag is not None
    dumped = json.dumps(dag.snapshot(), ensure_ascii=False)
    assert "судороги" not in dumped


# ---------------------------------------------------------------------------
# Part 2: batched single-round verifier contract.
# ---------------------------------------------------------------------------


def test_batched_schema_carries_verdicts_not_ids_or_aggregates() -> None:
    schema = verifier_batched_json_schema()
    assert schema["required"] == ["verdicts"]
    schema_properties = schema["properties"]
    assert isinstance(schema_properties, dict)
    verdicts_schema = schema_properties["verdicts"]
    assert isinstance(verdicts_schema, dict)
    assert verdicts_schema["type"] == "array"
    dumped = json.dumps(schema)
    assert "unit_id" not in dumped
    assert "all_required_supported" not in dumped
    assert '"scope"' not in dumped
    single = verifier_single_json_schema()
    single_properties = single["properties"]
    assert isinstance(single_properties, dict)
    assert set(single_properties) == {
        "requires_book_evidence",
        "supported",
        "evidence_passage_ids",
        "addresses_intent",
        "claim_origin",
        "origin_ref",
    }


def test_batched_decision_validation_requires_exact_count() -> None:
    good = _decision(book=True, supported=True, cites=["p1"])
    assert len(validate_batched_decisions({"verdicts": [good, good]}, expected_count=2)) == 2
    bad_payloads: tuple[object, ...] = (
        {"verdicts": [good]},
        {"verdicts": [good, good, good]},
        {"verdicts": []},
        {"verdicts": "nope"},
        {"nope": []},
        [good, good],
        {"verdicts": [{"supported": True}]},
        {"verdicts": [dict(good, supported="true")]},
    )
    for bad in bad_payloads:
        with pytest.raises(VerifierValidationError):
            validate_batched_decisions(bad, expected_count=2)


def test_batched_prompt_materializes_evidence_once_and_binds_by_order() -> None:
    pack = [_pack_entry(), _pack_entry(passage_id="chapter-3#exp0001", text="Второй отрывок.")]
    units = split_response_units(TEN_UNIT_DRAFT)
    assert len(units) == 10
    prompt, short_to_full, full_ids = build_batched_verifier_text(
        units=units, passages=pack, resolved_intent="намерение", user_message="сообщение"
    )
    for passage in pack:
        assert prompt.count(str(passage["text"])) == 1
    for pos in range(1, 11):
        assert f'pos="{pos}"' in prompt
    assert "unit_id" not in prompt
    assert "all_required_supported" not in prompt
    assert len(short_to_full) == 2
    assert set(full_ids) == {str(item["passage_id"]) for item in pack}
    suffix = batched_text_json_suffix(len(units))
    assert "10" in suffix


def test_coerce_batched_verdicts_matches_single_unit_strictness() -> None:
    pack = [_pack_entry()]
    units = split_response_units(TWO_UNIT_DRAFT)
    assert [unit.unit_id for unit in units] == ["u1", "u2"]
    verdicts = coerce_batched_verdicts(
        {
            "verdicts": [
                _decision(book=False, supported=True, cites=[]),
                _decision(
                    book=True,
                    supported=True,
                    cites=[pack[0]["passage_id"]],
                    origin="book_claim",
                ),
            ]
        },
        units,
        short_to_full={"p1": str(pack[0]["passage_id"])},
        full_ids={str(pack[0]["passage_id"])},
    )
    assert [item.unit_id for item in verdicts] == ["u1", "u2"]
    assert verdicts[0].scope != "book"
    assert verdicts[1].scope == "book"
    # Contradictory origin fails closed on the batched path too.
    with pytest.raises(VerifierValidationError):
        coerce_batched_verdicts(
            {
                "verdicts": [
                    _decision(book=False, supported=True, cites=[]),
                    _decision(
                        book=True,
                        supported=True,
                        cites=[pack[0]["passage_id"]],
                        origin="user_report",
                    ),
                ]
            },
            units,
        )


def test_parse_batched_text_decision_is_strict() -> None:
    good = _decision(book=True, supported=True, cites=["p1"])
    parsed = parse_batched_text_json_decision(
        json.dumps({"verdicts": [good, good], "extra": "dropped"}), expected_count=2
    )
    assert len(parsed["verdicts"]) == 2
    with pytest.raises(VerifierValidationError):
        parse_batched_text_json_decision(json.dumps({"verdicts": [good]}), expected_count=2)
    with pytest.raises(VerifierValidationError):
        parse_batched_text_json_decision("not json", expected_count=1)


async def test_batched_single_call_for_one_two_and_ten_units() -> None:
    pack = [_pack_entry()]
    passage_id = str(pack[0]["passage_id"])
    for draft, count in (
        (SINGLE_UNIT_DRAFT, 1),
        (TWO_UNIT_DRAFT, 2),
        (TEN_UNIT_DRAFT, 10),
    ):
        units = split_response_units(draft)
        assert len(units) == count
        decisions = [
            (
                _decision(book=False, supported=True, cites=[])
                if index == 0 and count > 1
                else _decision(book=True, supported=True, cites=[passage_id])
            )
            for index in range(count)
        ]
        model = _ScriptedAAVerifier([decisions], batched=True)
        result = await run_batched_verifier(
            units,
            pack,
            model=model,
            resolved_intent="поддержка",
            user_message=draft,
            answer_text=draft,
        )
        assert result.all_required_supported is True
        assert [item.unit_id for item in result.units] == [
            f"u{index + 1}" for index in range(count)
        ]
        assert model.calls == 1
        # The Evidence Pack travels once, not once per unit.
        assert model.prompts[0].count(str(pack[0]["text"])) == 1
        assert "verdicts" in json.dumps(model.schemas[0])


async def test_batched_missing_extra_and_invalid_verdicts_fail_closed() -> None:
    pack = [_pack_entry()]
    units = split_response_units(TWO_UNIT_DRAFT)
    good = _decision(book=True, supported=True, cites=[str(pack[0]["passage_id"])])
    # A structured-path validation failure retries once over the bounded
    # text path; the scripted text reply stays non-compliant, so every
    # ambiguous batch fails closed with exactly one structured call.
    garbage_text = "definitely not json{{{"
    for payload in (
        [good],
        [good, good, good],
        [dict(good, supported="true")],
        [dict(good, evidence_passage_ids=["no-such-passage"]), good],
        [dict(good, evidence_passage_ids=[])],
        [_decision(book=False, supported=True, cites=[str(pack[0]["passage_id"])])] * 2,
    ):
        model = _ScriptedAAVerifier([payload], batched=True, text_fallback=garbage_text)
        with pytest.raises(VerifierValidationError):
            await run_batched_verifier(units, pack, model=model, answer_text=TWO_UNIT_DRAFT)
        assert model.calls == 1


async def test_batched_poisoned_evidence_makes_no_model_call() -> None:
    pack = [_pack_entry()]
    pack[0]["text_sha256"] = "0" * 64
    units = split_response_units(SINGLE_UNIT_DRAFT)
    model = _ScriptedAAVerifier(
        [[_decision(book=True, supported=True, cites=[str(pack[0]["passage_id"])])]],
        batched=True,
    )
    with pytest.raises(VerifierValidationError):
        await run_batched_verifier(units, pack, model=model, answer_text=SINGLE_UNIT_DRAFT)
    assert model.calls == 0


async def test_batched_quote_integrity_over_full_source_range() -> None:
    exact = "Фиктивная поддержка рядом помогает пережить тягу"
    pack = [_pack_entry(text=exact + " спокойно и честно.")]
    good_draft = f"Вот точные слова: «{exact}»."
    good_units = split_response_units(good_draft)
    good_model = _ScriptedAAVerifier(
        [[_decision(book=True, supported=True, cites=[str(pack[0]["passage_id"])])]],
        batched=True,
    )
    good_result = await run_batched_verifier(
        good_units, pack, model=good_model, answer_text=good_draft
    )
    assert good_result.all_required_supported is True
    bad_draft = "Вот точные слова: «Фиктивная поддержка рядом помогает пережить всё»."
    bad_units = split_response_units(bad_draft)
    bad_model = _ScriptedAAVerifier(
        [[_decision(book=True, supported=True, cites=[str(pack[0]["passage_id"])])]],
        batched=True,
    )
    with pytest.raises(VerifierValidationError):
        await run_batched_verifier(bad_units, pack, model=bad_model, answer_text=bad_draft)


async def test_batched_429_propagates_with_single_call() -> None:
    pack = [_pack_entry()]
    units = split_response_units(TWO_UNIT_DRAFT)
    model = _ScriptedAAVerifier(
        [OpenCodeRateLimitError("opencode request rate-limited: http=429")], batched=True
    )
    with pytest.raises(OpenCodeRateLimitError):
        await run_batched_verifier(units, pack, model=model)
    assert model.calls == 1


async def test_batched_structured_fallback_serves_text_once() -> None:
    pack = [_pack_entry()]
    units = split_response_units(SINGLE_UNIT_DRAFT)
    reply = json.dumps(
        {"verdicts": [_decision(book=True, supported=True, cites=["p1"])]},
    )
    model = _ScriptedAAVerifier(
        [OpenCodeProviderAccessError("structured rejected")], batched=True, text_fallback=reply
    )
    result = await run_batched_verifier(units, pack, model=model, answer_text=SINGLE_UNIT_DRAFT)
    assert result.all_required_supported is True
    assert model.calls == 1
    assert model.text_calls == 1


async def test_run_verifier_auto_routes_by_capability() -> None:
    pack = [_pack_entry()]
    passage_id = str(pack[0]["passage_id"])
    units = split_response_units(TWO_UNIT_DRAFT)
    per_unit = _ScriptedAAVerifier(
        [
            _decision(book=False, supported=True, cites=[]),
            _decision(book=True, supported=True, cites=[passage_id]),
        ],
        batched=False,
    )
    assert model_supports_batched_verifier(per_unit) is False
    auto_per_unit = await run_verifier(units, pack, model=per_unit, answer_text=TWO_UNIT_DRAFT)
    assert auto_per_unit.all_required_supported is True
    assert per_unit.calls == 2
    batched = _ScriptedAAVerifier(
        [
            [
                _decision(book=False, supported=True, cites=[]),
                _decision(book=True, supported=True, cites=[passage_id]),
            ]
        ],
        batched=True,
    )
    assert model_supports_batched_verifier(batched) is True
    auto_batched = await run_verifier(units, pack, model=batched, answer_text=TWO_UNIT_DRAFT)
    assert auto_batched.all_required_supported is True
    assert batched.calls == 1
    legacy = _ScriptedAAVerifier(
        [
            [
                _decision(book=False, supported=True, cites=[]),
                _decision(book=True, supported=True, cites=[passage_id]),
            ]
        ],
        batched=True,
        advertise=False,
    )
    assert model_supports_batched_verifier(legacy) is False
    explicit = await run_verifier(
        units, pack, model=legacy, answer_text=TWO_UNIT_DRAFT, batch_mode="batched"
    )
    assert explicit.all_required_supported is True
    assert legacy.calls == 1
    with pytest.raises(VerifierValidationError):
        await run_verifier(units, pack, model=legacy, batch_mode="nope")


async def test_production_adapter_advertises_batched_capability() -> None:
    from aa.conversation.model_adapter import OpenCodeChatModel

    assert OpenCodeChatModel.supports_batched_verifier is True


# ---------------------------------------------------------------------------
# Part 3: production turn path on the batched round.
# ---------------------------------------------------------------------------


async def test_batched_substantive_turn_serves_verified_answer() -> None:
    pack = [_pack_entry()]
    passage_id = str(pack[0]["passage_id"])
    answer = _AnswerModel([TWO_UNIT_DRAFT])
    verifier = _ScriptedAAVerifier(
        [
            [
                _decision(book=False, supported=True, cites=[]),
                _decision(book=True, supported=True, cites=[passage_id]),
            ]
        ],
        batched=True,
    )
    outcome = await run_v2_answer_turn(
        user_message="Тянет выпить вечером, что делать?",
        summary="",
        recent=[HumanMessage(content="Здравствуйте")],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        initial_query_count=2,
    )
    assert outcome["text"] == TWO_UNIT_DRAFT
    assert verifier.calls == 1
    telemetry = outcome["telemetry"]
    assert telemetry["verifier_round_mode"] == "batched-single-round"
    assert telemetry["verifier_outcome"] == "passed"
    assert telemetry["qualified"] is True
    assert telemetry["semantic_work_plan"]["verifier_round_mode"] == "batched-single-round"
    dag = telemetry["semantic_dag"]
    assert re.fullmatch(r"[0-9a-f]{16}", str(dag.get("dag_id", ""))) is not None
    assert dag["verifier_units"] == 2
    assert dag["verifier_materialized_chars"] == dag["verifier_evidence_chars"]
    assert telemetry["provider_attempts"] >= 2
    dumped = json.dumps(telemetry, ensure_ascii=False)
    assert "Тянет выпить" not in dumped
    assert str(pack[0]["text"]) not in dumped


async def test_batched_mixed_verdicts_narrow_to_supported_text() -> None:
    pack = [_pack_entry()]
    passage_id = str(pack[0]["passage_id"])
    draft = "Поддержка рядом помогает пережить тягу. А ещё помогает выпить немного."
    units = split_response_units(draft)
    assert len(units) == 2
    answer = _AnswerModel([draft])
    verifier = _ScriptedAAVerifier(
        [
            [
                _decision(book=True, supported=True, cites=[passage_id]),
                _decision(book=True, supported=False, cites=[passage_id]),
            ]
        ],
        batched=True,
    )
    outcome = await run_v2_answer_turn(
        user_message="Тянет выпить вечером, что делать?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        initial_query_count=2,
    )
    assert outcome["text"] == units[0].text
    assert verifier.calls == 1
    assert outcome["telemetry"]["verifier_unavailable_units"] == 0


async def test_batched_glue_turn_passes_without_book() -> None:
    draft = "Понимаю вас."
    assert len(split_response_units(draft)) == 1
    answer = _AnswerModel([draft])
    verifier = _ScriptedAAVerifier(
        [[_decision(book=False, supported=True, cites=[])]], batched=True
    )
    outcome = await run_v2_answer_turn(
        user_message="Спасибо за поддержку…",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=answer,
        verifier_model=verifier,
    )
    assert outcome["text"] == draft
    assert verifier.calls == 1


async def test_batched_out_of_book_claim_fails_closed() -> None:
    pack = [_pack_entry()]
    draft = "Завтра будет дождь, возьмите зонт."
    assert len(split_response_units(draft)) == 1
    answer = _AnswerModel([draft])
    verifier = _ScriptedAAVerifier(
        [[_decision(book=True, supported=False, cites=[])]], batched=True
    )
    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(
            user_message="Какая завтра погода?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=answer,
            verifier_model=verifier,
            initial_query_count=2,
        )
    assert exc.value.telemetry["qualified"] is False
    assert exc.value.telemetry["verifier_outcome"] == "unsupported"
    assert exc.value.telemetry["verifier_round_mode"] == "batched-single-round"


async def test_batched_429_propagates_end_to_end() -> None:
    pack = [_pack_entry()]
    answer = _AnswerModel([SINGLE_UNIT_DRAFT])
    verifier = _ScriptedAAVerifier(
        [OpenCodeRateLimitError("opencode request rate-limited: http=429")], batched=True
    )
    with pytest.raises(OpenCodeRateLimitError):
        await run_v2_answer_turn(
            user_message="Тянет выпить вечером, что делать?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=answer,
            verifier_model=verifier,
            initial_query_count=2,
        )
    assert verifier.calls == 1


async def test_verified_decision_reuse_skips_second_batched_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.conversation.retrieval_node as retrieval_node_mod

    pack = [_pack_entry()]
    passage_id = str(pack[0]["passage_id"])
    bad = "Выпить немного помогает расслабиться. А ещё помогает выпить побольше."
    assert len(split_response_units(bad)) == 2
    answer = _AnswerModel([bad, bad])
    verifier = _ScriptedAAVerifier(
        [
            [
                _decision(book=True, supported=False, cites=[passage_id]),
                _decision(book=True, supported=False, cites=[passage_id]),
            ]
        ],
        batched=True,
    )
    planner = _PlannerModel(
        [
            {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries("луна"),
            }
        ]
    )

    async def _fake_retrieve(
        index: Any, queries: object, *, config: Any = None, **kwargs: Any
    ) -> Any:
        from aa.retrieval.evidence import EvidencePack

        _ = (index, queries, config, kwargs)
        return EvidencePack(passages=(), total_tokens=0, corpus_version="v", retrieval_metadata={})

    monkeypatch.setattr(retrieval_node_mod, "aretrieve_with_semantic_selection", _fake_retrieve)
    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(
            user_message="Тянет выпить вечером, что делать?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=answer,
            verifier_model=verifier,
            planner_model=planner,
            retrieval_index=object(),
            max_repair_rounds=2,
        )
    # The identical regen against the unchanged pack reuses the cached
    # verdicts: exactly one batched provider call for two verifications.
    assert verifier.calls == 1
    assert exc.value.telemetry["verifier_cache_hits"] == 1


async def test_no_full_pipeline_repeat_on_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.conversation.retrieval_node as retrieval_node_mod

    pack = [_pack_entry()]
    passage_id = str(pack[0]["passage_id"])
    bad_first = "Выпить немного помогает расслабиться. А ещё помогает выпить побольше."
    bad_second = "Немного выпить снимает напряжение. А побольше помогает ещё лучше."
    assert len(split_response_units(bad_first)) == 2
    assert len(split_response_units(bad_second)) == 2
    answer = _AnswerModel([bad_first, bad_second, bad_second])
    verifier = _ScriptedAAVerifier(
        [
            [
                _decision(book=True, supported=False, cites=[passage_id]),
                _decision(book=True, supported=False, cites=[passage_id]),
            ],
            [
                _decision(book=True, supported=False, cites=[passage_id]),
                _decision(book=True, supported=False, cites=[passage_id]),
            ],
        ],
        batched=True,
    )
    planner = _PlannerModel(
        [
            {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries("луна"),
            },
            {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries("вечер"),
            },
        ]
    )
    retrieve_calls = 0
    extra_text = "Фиктивная другая поддержка рядом."
    extra_sha = hashlib.sha256(extra_text.encode()).hexdigest()

    async def _fake_retrieve(
        index: Any, queries: object, *, config: Any = None, **kwargs: Any
    ) -> Any:
        from aa.retrieval.evidence import EvidencePack, EvidencePassageData

        nonlocal retrieve_calls
        _ = (index, queries, config, kwargs)
        retrieve_calls += 1
        passage = EvidencePassageData(
            passage_id="chapter-3#extra-stall",
            exact_text=extra_text,
            source_id="ru-fourth-edition-txt",
            section_id="chapter-3",
            child_chunk_ids=("chapter-3:ru:9",),
            char_start=999,
            char_end=1030,
            text_sha256=extra_sha,
            source_sha256="s" * 64,
        )
        return EvidencePack(
            passages=(passage,), total_tokens=10, corpus_version="v", retrieval_metadata={}
        )

    monkeypatch.setattr(retrieval_node_mod, "aretrieve_with_semantic_selection", _fake_retrieve)
    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(
            user_message="Тянет выпить вечером, что делать?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=answer,
            verifier_model=verifier,
            planner_model=planner,
            retrieval_index=object(),
            max_repair_rounds=2,
        )
    # Round one grows the pack; round two retrieves nothing new, so the
    # loop stops instead of repeating the full pipeline: exactly two
    # repair retrievals, one repair round, no third grind.
    assert exc.value.telemetry["repair_rounds"] == 1
    assert exc.value.telemetry["qualified"] is False
    assert retrieve_calls == 2
    assert verifier.calls == 2


# ---------------------------------------------------------------------------
# Part 4: load behavior: slow/blocked providers, 429, concurrency, cleanup.
# ---------------------------------------------------------------------------


class _SlowBatchedVerifier:
    supports_batched_verifier = True

    def __init__(self, delay_s: float, decisions: list[dict[str, Any]]) -> None:
        self._delay_s = delay_s
        self._decisions = decisions
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        await asyncio.sleep(self._delay_s)
        return {"verdicts": [dict(item) for item in self._decisions]}


class _ResistantBatchedVerifier(_SlowBatchedVerifier):
    """Cancellation-resistant provider: absorbs one cancel, then complies."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        try:
            await asyncio.sleep(self._delay_s)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            raise
        return {"verdicts": [dict(item) for item in self._decisions]}


async def test_slow_provider_hard_deadline_with_cleanup() -> None:
    pack = [_pack_entry()]
    units = split_response_units(SINGLE_UNIT_DRAFT)
    model = _SlowBatchedVerifier(
        30.0, [_decision(book=True, supported=True, cites=[str(pack[0]["passage_id"])])]
    )
    before = set(asyncio.all_tasks())
    started = time.perf_counter()
    with pytest.raises(OpenCodeTimeoutError):
        await run_batched_verifier(units, pack, model=model, turn_budget_s=0.3)
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0
    assert model.calls == 1
    for _ in range(10):
        await asyncio.sleep(0)
    leftover = {
        task
        for task in set(asyncio.all_tasks()) - before
        if task is not asyncio.current_task() and not task.done()
    }
    assert leftover == set()


async def test_cancellation_resistant_provider_still_bounded() -> None:
    pack = [_pack_entry()]
    units = split_response_units(SINGLE_UNIT_DRAFT)
    model = _ResistantBatchedVerifier(
        30.0, [_decision(book=True, supported=True, cites=[str(pack[0]["passage_id"])])]
    )
    started = time.perf_counter()
    with pytest.raises(OpenCodeTimeoutError):
        await run_batched_verifier(units, pack, model=model, turn_budget_s=0.3)
    assert time.perf_counter() - started < 5.0
    assert model.calls == 1


async def test_concurrent_turns_share_no_state_and_leave_no_orphans() -> None:
    pack = [_pack_entry()]
    passage_id = str(pack[0]["passage_id"])

    async def _one_turn(index: int) -> dict[str, Any]:
        _ = index
        answer = _AnswerModel([SINGLE_UNIT_DRAFT])
        verifier = _ScriptedAAVerifier(
            [[_decision(book=True, supported=True, cites=[passage_id])]], batched=True
        )
        await asyncio.sleep(0.05)
        return await run_v2_answer_turn(
            user_message="Тянет выпить вечером, что делать?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=answer,
            verifier_model=verifier,
            initial_query_count=2,
        )

    before = set(asyncio.all_tasks())
    started = time.perf_counter()
    outcomes = await asyncio.gather(*[_one_turn(index) for index in range(6)])
    elapsed = time.perf_counter() - started
    assert elapsed < 15.0
    dag_ids = set()
    for outcome in outcomes:
        assert outcome["text"] == SINGLE_UNIT_DRAFT
        telemetry = outcome["telemetry"]
        assert telemetry["qualified"] is True
        dag_ids.add(str(telemetry["semantic_dag"]["dag_id"]))
    assert len(dag_ids) == 6
    for _ in range(10):
        await asyncio.sleep(0)
    leftover = {
        task
        for task in set(asyncio.all_tasks()) - before
        if task is not asyncio.current_task() and not task.done()
    }
    assert leftover == set()


async def test_concurrent_429_turn_does_not_sink_healthy_siblings() -> None:
    pack = [_pack_entry()]
    passage_id = str(pack[0]["passage_id"])

    async def _healthy_turn() -> str:
        outcome = await run_v2_answer_turn(
            user_message="Тянет выпить вечером, что делать?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=_AnswerModel([SINGLE_UNIT_DRAFT]),
            verifier_model=_ScriptedAAVerifier(
                [[_decision(book=True, supported=True, cites=[passage_id])]], batched=True
            ),
            initial_query_count=2,
        )
        return str(outcome["text"])

    async def _limited_turn() -> str:
        with pytest.raises(OpenCodeRateLimitError):
            await run_v2_answer_turn(
                user_message="Тянет выпить вечером, что делать?",
                summary="",
                recent=[],
                evidence_pack=pack,
                answer_model=_AnswerModel([SINGLE_UNIT_DRAFT]),
                verifier_model=_ScriptedAAVerifier(
                    [OpenCodeRateLimitError("opencode request rate-limited: http=429")],
                    batched=True,
                ),
                initial_query_count=2,
            )
        return "rate-limited"

    results = await asyncio.gather(*[_healthy_turn() for _ in range(3)], _limited_turn())
    assert results[:3] == [SINGLE_UNIT_DRAFT] * 3
    assert results[3] == "rate-limited"


async def test_slow_answer_turn_fails_fast_inside_internal_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.conversation.turn_pipeline as pipeline

    monkeypatch.setattr(pipeline, "ANSWER_DRAFT_ATTEMPT_BUDGET_S", 1.0)
    pack = [_pack_entry()]

    class _SlowAnswer:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            self.calls += 1
            await asyncio.sleep(30.0)
            return AIMessage(content=SINGLE_UNIT_DRAFT)

    answer = _SlowAnswer()
    verifier = _ScriptedAAVerifier(
        [[_decision(book=True, supported=True, cites=[str(pack[0]["passage_id"])])]],
        batched=True,
    )
    started = time.perf_counter()
    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(
            user_message="Тянет выпить вечером, что делать?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=answer,
            verifier_model=verifier,
            initial_query_count=2,
        )
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0
    assert answer.calls == 1
    assert verifier.calls == 0
    assert exc.value.telemetry["turn_budget_exceeded"] is False
    assert exc.value.telemetry["semantic_dag"]["dag_id"]
