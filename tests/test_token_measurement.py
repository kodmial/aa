"""Tests for measured RU/EN context-cost accounting (issue #49, offline).

Every test here is hermetic: token payloads are canned, fixtures are
local, and no model or network access is required. Live runtime numbers
are recorded separately in ``docs/context-cost-measurement.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from aa.corpus.token_measurement import (
    ALIGNED_PAIRS,
    MEASURE_INSTRUCTION,
    RUSSIAN_HISTORIES,
    CaseResult,
    build_turn,
    collect_case,
    describe_case,
    effective_input_tokens,
    parse_token_usage,
    summarize_runs,
)


def _payload(
    input_tokens: int = 100,
    output_tokens: int = 11,
    reasoning_tokens: int = 7,
    cache_read: int = 50,
    cache_write: int = 0,
) -> dict[str, Any]:
    return {
        "input": input_tokens,
        "output": output_tokens,
        "reasoning": reasoning_tokens,
        "cache": {"read": cache_read, "write": cache_write},
    }


def test_parse_token_usage_extracts_all_fields() -> None:
    usage = parse_token_usage(_payload())
    assert usage.input_tokens == 100
    assert usage.output_tokens == 11
    assert usage.reasoning_tokens == 7
    assert usage.cache_read_tokens == 50
    assert usage.cache_write_tokens == 0


def test_effective_input_uses_stable_sum_across_cache_splits() -> None:
    # The provider splits identical prompt costs between ``input`` and
    # ``cache.read`` nondeterministically; the sum must be identical.
    first = parse_token_usage(_payload(input_tokens=559, cache_read=7793))
    second = parse_token_usage(_payload(input_tokens=47, cache_read=8305))
    assert effective_input_tokens(first) == effective_input_tokens(second) == 8352


def test_parse_token_usage_rejects_malformed_payloads() -> None:
    with pytest.raises(ValueError):
        parse_token_usage({})
    with pytest.raises(ValueError):
        parse_token_usage({"input": 1, "output": 1, "reasoning": 1})
    with pytest.raises(ValueError):
        parse_token_usage(_payload(input_tokens=-1))
    with pytest.raises(ValueError):
        parse_token_usage(_payload(cache_read=None))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parse_token_usage(_payload(input_tokens="100"))  # type: ignore[arg-type]


def test_summarize_runs_median_and_p95() -> None:
    stats = summarize_runs([10, 30, 20])
    assert stats.count == 3
    assert stats.minimum == 10
    assert stats.median == 20.0
    assert stats.maximum == 30
    assert stats.p95 == 30


def test_summarize_runs_even_count_median() -> None:
    stats = summarize_runs([10, 20])
    assert stats.median == 15.0
    assert stats.p95 == 20


def test_summarize_runs_single_value() -> None:
    stats = summarize_runs([42])
    assert (stats.minimum, stats.median, stats.p95, stats.maximum) == (42, 42.0, 42, 42)


def test_summarize_runs_rejects_empty_and_invalid() -> None:
    with pytest.raises(ValueError):
        summarize_runs([])
    with pytest.raises(ValueError):
        summarize_runs([10, -1])


def test_collect_case_returns_effective_costs_per_repeat() -> None:
    seen: list[str] = []

    def fake_send(payload: str) -> Mapping[str, Any]:
        seen.append(payload)
        return _payload(input_tokens=100, cache_read=44)

    costs = collect_case(fake_send, "probe", repeats=3)
    assert costs == [144, 144, 144]
    assert seen == ["probe"] * 3


def test_collect_case_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError):
        collect_case(lambda payload: _payload(), "probe", repeats=0)


def test_describe_case_delta_and_chars_per_token() -> None:
    result = describe_case(
        name="map_ru",
        language="ru",
        payload="x" * 512,
        costs=[100, 100, 100],
        baseline_median=40.0,
    )
    assert isinstance(result, CaseResult)
    assert result.delta_tokens == 60.0
    assert result.chars_per_token == pytest.approx(512 / 60.0)
    assert result.stats.median == 100.0


def test_describe_case_no_per_token_when_no_positive_delta() -> None:
    result = describe_case(
        name="baseline",
        language="mixed",
        payload="abc",
        costs=[40, 40],
        baseline_median=40.0,
    )
    assert result.delta_tokens == 0.0
    assert result.chars_per_token is None


def test_aligned_pairs_share_markers_and_differ_by_language() -> None:
    assert set(ALIGNED_PAIRS) == {"book_map", "evidence_pack", "planner_wrapper", "system_sample"}
    for name, pair in ALIGNED_PAIRS.items():
        en, ru = pair["en"], pair["ru"]
        assert en.strip() and ru.strip(), name
        assert en != ru, name
    # Section ids must match across the book-map pair.
    for section in ("doctors-opinion", "chapter-1", "chapter-5"):
        assert section in ALIGNED_PAIRS["book_map"]["en"]
        assert section in ALIGNED_PAIRS["book_map"]["ru"]
    # Passage ids must match across the evidence pair.
    for passage in ("[E1 ch3]", "[E2 ch5]"):
        assert passage in ALIGNED_PAIRS["evidence_pack"]["en"]
        assert passage in ALIGNED_PAIRS["evidence_pack"]["ru"]
    # Tool names and chunk ids must match across the wrapper pair.
    for marker in ("book_search", "book_read", "book_section", "ch3-p12", "chapter-8"):
        assert marker in ALIGNED_PAIRS["planner_wrapper"]["en"]
        assert marker in ALIGNED_PAIRS["planner_wrapper"]["ru"]
    # The RU sides must carry Cyrillic payload (otherwise no RU cost is measured).
    for name, pair in ALIGNED_PAIRS.items():
        assert any("\u0400" <= ch <= "\u04ff" for ch in pair["ru"]), name
        assert not any("\u0400" <= ch <= "\u04ff" for ch in pair["en"]), name


def test_planner_wrappers_are_valid_json_with_same_keys() -> None:
    def plan(text: str) -> Any:
        start = text.index("{")
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        return parsed

    en_plan = plan(ALIGNED_PAIRS["planner_wrapper"]["en"])
    ru_plan = plan(ALIGNED_PAIRS["planner_wrapper"]["ru"])
    assert [item["tool"] for item in en_plan["plan"]] == [item["tool"] for item in ru_plan["plan"]]
    assert [item.get("chunk", item.get("section")) for item in en_plan["plan"]] == [
        item.get("chunk", item.get("section")) for item in ru_plan["plan"]
    ]


def test_russian_histories_grow_with_length_and_carry_cyrillic() -> None:
    assert set(RUSSIAN_HISTORIES) == {"history_short", "history_medium", "history_long"}
    short = RUSSIAN_HISTORIES["history_short"]
    medium = RUSSIAN_HISTORIES["history_medium"]
    long = RUSSIAN_HISTORIES["history_long"]
    assert len(short) < len(medium) < len(long)
    for name, history in RUSSIAN_HISTORIES.items():
        assert any("\u0400" <= ch <= "\u04ff" for ch in history), name


def test_build_turn_prefixes_instruction_and_joins_parts() -> None:
    turn = build_turn("HISTORY", "MAP")
    assert turn.startswith(MEASURE_INSTRUCTION)
    assert "HISTORY" in turn and "MAP" in turn
