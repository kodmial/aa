"""Production grounded conversational runtime tests (issue #9).

Uses invented fixture sentences only; no canonical book text is committed.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
import re
import subprocess
import sys
from typing import Any

import pytest

from aa.conversation.graph_runtime import GraphTurnRuntime
from aa.conversation.orchestrator import (
    AGENT_NAME,
    FAIL_CLOSED_REPLY,
    RUNTIME_VERSION,
    SUPPORT_SCHEMA_VERSION,
    TurnFailed,
    TurnRunner,
    build_grounded_response,
    build_local_plan_payload,
    build_synthesis_prompt,
    deduplicate_cross_aspect,
    fit_evidence_budget,
    is_substantive,
    judge_unit,
    load_exact_evidence,
    run_planner,
    run_trivial_turn,
    search_first_round,
    send_with_fallback,
    split_answer_units,
    validate_coverage_payload,
    validate_support_payload,
)
from aa.corpus.structure import SECTION_IDS, build_full_structure
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeRateLimitError,
    OpenCodeTransientError,
)
from aa.retrieval.index import HybridIndex, build_hybrid_index
from aa.retrieval.planner import aspect_search_queries, validate_plan

PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"


def test_graph_runtime_preserves_stage_latency_and_outage_telemetry() -> None:
    runtime = GraphTurnRuntime()
    result = {
        "retry_state": {
            "turn_telemetry": {
                "planner_outcome": "ok",
                "planner_latency_ms": 1200.0,
                "retrieval_outcome": "evidence-ready",
                "retrieval_latency_ms": 250.0,
                "answer_outcome": "narrowed-supported",
                "answer_latency_ms": 6400.0,
                "answer_rounds": 1,
                "verifier_outcome": "partial-unavailable",
                "verifier_latency_ms": 3100.0,
                "verifier_unavailable_units": 1,
                "repair_rounds": 0,
                "repair_budget_exceeded": False,
            }
        },
        "search_queries": ["q"] * 10,
        "evidence_pack": [{"passage_id": "p1"}],
        "grounding_result": {"all_required_supported": False},
    }
    runtime._record_stage_telemetry("thread", result, 11000.0, 42)
    snapshot = runtime.last_telemetry_for_thread("thread")
    assert snapshot["answer_latency_ms"] == 6400.0
    assert snapshot["verifier_latency_ms"] == 3100.0
    assert snapshot["answer_rounds"] == 1
    assert snapshot["verifier_unavailable_units"] == 1
    assert snapshot["repair_budget_exceeded"] is False



RU_FIXTURES: dict[str, str] = {
    "doctors-opinion": (
        "Фиктивное мнение доктора о тяге и навязчивом желании выпить. "
        "Тяга приходит внезапно и требует внимания.\n\n"
        "Второй абзац мнения доктора. Наблюдение за тягой продолжается."
    ),
    "chapter-1": (
        "Фиктивный рассказ о первом глотке. Герой начал бухать каждый вечер "
        "и однажды тяпнул лишнего. Утром после выпивки было тяжело.\n\n"
        "Второй абзац рассказа. Нажрался в гостях и долго приходил в себя."
    ),
    "chapter-2": (
        "Фиктивный выход есть для пьющих. Надежда и поддержка рядом. "
        "Пьющие находят помощь в сообществе.\n\n"
        "Второй абзац выхода. Сообщество встречает новичков тепло."
    ),
    "chapter-3": (
        "Фиктивный алкоголизм как феномен тяги. Тяга к алкоголю и последствия "
        "алкоголизма обсуждаются подробно. Сорвался после долгой трезвости.\n\n"
        "Второй абзац об алкоголизме. Тянет выпить снова, страх срыва рядом."
    ),
    "chapter-4": (
        "Фиктивные размышления агностика. Готовность принять помощь растет.\n\n"
        "Второй абзац агностика. Сомнения обсуждаются открыто."
    ),
    "chapter-5": (
        "Фиктивная программа в действии требует честности. "
        "Практические шаги каждый день.\n\n"
        "Второй абзац программы. Утренний настрой задает тон."
    ),
    "chapter-6": (
        "Фиктивная работа по шагам продолжается. Утром делаем инвентаризацию. "
        "Срыв разбираем честно и спокойно.\n\n"
        "Второй абзац работы. Вечером подводим итоги дня."
    ),
    "chapter-7": (
        "Фиктивная работа с другими людьми. Несем весть тем кто страдает.\n\n"
        "Второй абзац помощи. Разговор ведется спокойно и честно."
    ),
    "chapter-8": (
        "Фиктивная жинка ругает из-за пьянки. Женушка переживает за семью. "
        "Жена ругает пьянство, доверие страдает.\n\n"
        "Второй абзац о семье. Пьянство разрушает доверие постепенно."
    ),
    "chapter-9": (
        "Фиктивные новые отношения в семье. Доверие возвращается постепенно. "
        "Семейный конфликт утихает.\n\n"
        "Второй абзац семьи. Разговоры становятся спокойнее."
    ),
    "chapter-10": (
        "Фиктивное обращение к работодателям. Трезвость на рабочем месте важна. "
        "Рабочий конфликт и страх увольнения обсуждаются.\n\n"
        "Второй абзац работодателям. Поддержка коллег помогает многим."
    ),
    "chapter-11": (
        "Фиктивный взгляд в будущее сообщества. Бухать больше не хочется, "
        "хочется жить трезво.\n\n"
        "Второй абзац будущего. Планы строятся на трезвую голову."
    ),
}


def _fixture_index(tmp_path: pathlib.Path) -> HybridIndex:
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": f"Fixture EN {section_id} opening. Second sentence here.\n\n"
                f"Fixture EN {section_id} second paragraph.",
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": hashlib.sha256(b"en-source").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU TITLE {section_id}",
                "text": RU_FIXTURES[section_id],
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": hashlib.sha256(b"ru-source").hexdigest(),
            }
        )
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    out_dir = tmp_path / "retrieval"
    return build_hybrid_index(
        dict(full),
        ru_manifest={"format": "x", "artifact_sha256": "a" * 64},
        en_manifest={"format": "x", "artifact_sha256": "b" * 64},
        embedding_lock={"model_id": "m", "revision": "r" * 40},
        out_dir=out_dir,
        backend="hashing",
    )


def _runner(index: HybridIndex) -> TurnRunner:
    return TurnRunner(
        index=index,
        ru_corpus_version="",
        agent=AGENT_NAME,
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )


async def _noop_sleep(_delay: float) -> None:
    return None


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (single grounding unit)."""
    parts = [item.strip() for item in re.split(r"(?<=[.!?…])\s+", text) if item.strip()]
    return parts[0] if parts else text.strip()


def _grounded_send_from_prompt(prompt: str) -> str:
    """Deterministic stub synthesis: quote the first evidence passage."""
    for line in prompt.splitlines():
        match = re.match(r"\[([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+#[A-Za-z0-9_:.\-]+)\] (.*)", line)
        if match:
            locator, text = match.group(1), match.group(2).strip()
            return f"{_first_sentence(text)} [{locator}]"
    raise AssertionError("synthesis prompt carries no evidence")


# ---------------------------------------------------------------------------
# Substantive classification + planner behavior
# ---------------------------------------------------------------------------


def test_trivial_greetings_are_not_substantive() -> None:
    for trivial in ("привет", "здравствуйте", "hello", "спасибо", "/start", "   "):
        assert is_substantive(trivial) is False


def test_substantive_turns_need_the_full_pipeline() -> None:
    for substantive in (
        "я бухаю каждый вечер, что делать",
        "жинка ругает из-за пьянки",
        "я сорвался после долгой трезвости",
        "What does the Big Book say about fear?",
        "Что говорит Большая книга о страхе?",
    ):
        assert is_substantive(substantive) is True


def test_local_planner_preserves_original_and_expands_slang() -> None:
    text = "я бухаю каждый вечер"
    payload = build_local_plan_payload(text, utterance_id="u1")
    plan = validate_plan(payload, original_query=text)
    assert plan.original_query == text
    assert plan.language == "ru"
    fused: list[str] = []
    for aspect in plan.aspects:
        fused.extend(aspect_search_queries(aspect, original_query=text))
    assert fused[0] == text
    assert plan.aspects[0].lexical_query_en is None
    haystack = " ".join(fused).casefold()
    assert "употребляю алкоголь" in haystack or "выпивка" in haystack


def test_planner_handles_family_colloquialism_without_strengthening() -> None:
    text = "жинка ругает из-за пьянки"
    plan = validate_plan(build_local_plan_payload(text), original_query=text)
    haystack = " ".join(
        [*plan.aspects[0].lexical_queries_ru, *plan.aspects[0].semantic_queries_ru]
    ).casefold()
    assert "жена" in haystack or "семья" in haystack
    assert "развод" not in haystack
    assert "насили" not in haystack


def test_planner_marks_sorvalsya_ambiguity_as_material() -> None:
    plan = validate_plan(build_local_plan_payload("я сорвался"), original_query="я сорвался")
    assert plan.aspects[0].ambiguity == "material"


def test_planner_keeps_typos_and_adds_useful_rewrites() -> None:
    text = "жнка ругает изза пянки"
    plan = validate_plan(build_local_plan_payload(text), original_query=text)
    assert plan.original_query == text
    haystack = " ".join(
        [*plan.aspects[0].lexical_queries_ru, *plan.aspects[0].semantic_queries_ru]
    ).casefold()
    assert "жена" in haystack or "семья" in haystack


def test_broad_request_plans_multiple_aspects() -> None:
    text = "я бухаю, жинка ругает и на работе проблемы"
    plan = validate_plan(build_local_plan_payload(text), original_query=text)
    assert len(plan.aspects) >= 2


async def test_malformed_planner_output_retries_once_then_fails_closed() -> None:
    calls = 0

    def _bad_planner(_text: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"schema_version": "wrong", "aspects": []}

    with pytest.raises(TurnFailed):
        await run_planner("я бухаю", planner_fn=_bad_planner)
    assert calls == 2


async def test_en_branch_in_planner_fails_closed() -> None:
    def _en_planner(_text: str) -> dict[str, Any]:
        payload = build_local_plan_payload("я бухаю")
        aspect = dict(payload["aspects"][0])
        aspect["lexical_query_en"] = "drinking"
        payload["aspects"] = [aspect]
        return payload

    with pytest.raises(TurnFailed):
        await run_planner("я бухаю", planner_fn=_en_planner)


# ---------------------------------------------------------------------------
# RU-first retrieval / evidence enforcement
# ---------------------------------------------------------------------------


async def test_ru_first_search_reads_exact_ru_evidence(tmp_path: pathlib.Path) -> None:
    index = _fixture_index(tmp_path)
    plan = await run_planner("я бухаю каждый вечер")
    hits = search_first_round(index, plan)
    assert hits
    merged = deduplicate_cross_aspect(hits)
    assert merged
    assert all(":ru:" in hit.chunk_id for hit in merged)
    pack, _calls = load_exact_evidence(
        index,
        merged,
        ru_corpus_version=str(index.metadata.get("ru_artifact_sha256", "")),
    )
    assert pack.units
    assert all(unit.language == "ru" for unit in pack.units)
    assert all(unit.is_source_text for unit in pack.units)
    for unit in pack.units:
        unit.provenance.validate()


def test_coverage_payload_is_validated() -> None:
    ok = validate_coverage_payload(
        {
            "schema_version": "aa-coverage-v1",
            "covered": True,
            "gaps": [],
            "distinct_sections": 3,
            "aspects_covered": 2,
            "aspects_total": 2,
        }
    )
    assert ok.covered is True
    with pytest.raises(TurnFailed):
        validate_coverage_payload({"schema_version": "aa-coverage-v1", "covered": "yes"})


def test_support_payload_is_validated() -> None:
    units = validate_support_payload(
        {
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "units": [
                {
                    "unit_id": "u1",
                    "claim": "трезвость рядом",
                    "cited": ["ru-fourth-edition-txt/chapter-2#c1"],
                    "quote_kind": "exact-source",
                }
            ],
        }
    )
    assert units[0]["unit_id"] == "u1"
    with pytest.raises(TurnFailed):
        validate_support_payload({"schema_version": SUPPORT_SCHEMA_VERSION, "units": []})


async def test_evidence_budget_keeps_atomic_prefix(tmp_path: pathlib.Path) -> None:
    index = _fixture_index(tmp_path)
    plan = await run_planner("тяга и страх срыва")
    merged = deduplicate_cross_aspect(search_first_round(index, plan))
    pack, _ = load_exact_evidence(
        index, merged, ru_corpus_version=str(index.metadata.get("ru_artifact_sha256", ""))
    )
    fitted = fit_evidence_budget(pack, budget_tokens=10**9)
    assert len(fitted.units) == len(pack.units)
    tiny = fit_evidence_budget(pack, budget_tokens=10**9)
    assert tiny.token_count == pack.token_count


async def test_stale_version_read_fails_closed(tmp_path: pathlib.Path) -> None:
    index = _fixture_index(tmp_path)
    plan = await run_planner("я бухаю")
    merged = deduplicate_cross_aspect(search_first_round(index, plan))
    assert merged
    from aa.retrieval.book_tools import BookStaleError, book_read

    with pytest.raises(BookStaleError):
        book_read(index, merged[0].logical_chunk_id, expected_ru_version="0" * 64)


# ---------------------------------------------------------------------------
# Full state machine
# ---------------------------------------------------------------------------


async def test_full_grounded_turn_is_exact_and_supported(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    index = _fixture_index(tmp_path)
    runner = _runner(index)
    secret = "я бухаю каждый вечер xyzzy-secret"

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        assert agent == AGENT_NAME
        assert model == PRIMARY
        assert session_id == "ses_1"
        return _grounded_send_from_prompt(prompt)

    with caplog.at_level(logging.INFO, logger="aa.conversation.orchestrator"):
        response = await runner.run_grounded_turn(secret, session_id="ses_1", send=_send)
    assert response.diagnostics.grounding_passed is True
    assert response.diagnostics.regeneration_count == 0
    assert response.diagnostics.served_model == PRIMARY
    assert response.evidence is not None
    assert response.diagnostics.retrieval_rounds in (1, 2)
    substantive = [unit for unit in response.units if unit.grounding_passed is not None]
    assert substantive
    assert all(unit.grounding_passed for unit in substantive)
    assert any(unit.source_exact for unit in substantive)
    for record in caplog.records:
        assert "xyzzy-secret" not in record.getMessage()


async def test_unsupported_synthesis_regenerates_once_then_passes(
    tmp_path: pathlib.Path,
) -> None:
    index = _fixture_index(tmp_path)
    runner = _runner(index)
    calls = 0

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "Выдуманное утверждение про несуществующие обстоятельства"
        return _grounded_send_from_prompt(prompt)

    response = await runner.run_grounded_turn(
        "я бухаю каждый вечер", session_id="ses_1", send=_send
    )
    assert calls == 2
    assert response.diagnostics.regeneration_count == 1
    assert response.diagnostics.grounding_passed is True


async def test_persistently_unsupported_answer_fails_closed(tmp_path: pathlib.Path) -> None:
    index = _fixture_index(tmp_path)
    runner = _runner(index)

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        return "Совершенно посторонняя заметка про огородные работы зимой"

    with pytest.raises(TurnFailed) as excinfo:
        await runner.run_grounded_turn("я бухаю", session_id="ses_1", send=_send)
    assert excinfo.value.category in ("grounding-failed", "synthesis-empty")


async def test_missing_corpus_fails_closed() -> None:
    runner = TurnRunner(
        index=None, agent=AGENT_NAME, primary_model=PRIMARY, fallback_model=FALLBACK
    )

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        raise AssertionError("synthesis must never run without RU evidence")

    with pytest.raises(TurnFailed) as excinfo:
        await runner.run_grounded_turn("я бухаю", session_id="ses_1", send=_send)
    assert excinfo.value.category == "corpus-unavailable"


async def test_stale_pinned_version_fails_closed(tmp_path: pathlib.Path) -> None:
    index = _fixture_index(tmp_path)
    runner = TurnRunner(
        index=index,
        ru_corpus_version="0" * 64,
        agent=AGENT_NAME,
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
    )

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        raise AssertionError("synthesis must never run on a stale corpus")

    with pytest.raises(TurnFailed) as excinfo:
        await runner.run_grounded_turn("я бухаю", session_id="ses_1", send=_send)
    assert excinfo.value.category == "corpus-stale"


async def test_source_id_alone_cannot_ground(tmp_path: pathlib.Path) -> None:
    index = _fixture_index(tmp_path)
    plan = await run_planner("я бухаю")
    merged = deduplicate_cross_aspect(search_first_round(index, plan))
    pack, _ = load_exact_evidence(
        index, merged, ru_corpus_version=str(index.metadata.get("ru_artifact_sha256", ""))
    )
    units = split_answer_units("Фиктивное утверждение без ссылок на отрывки")
    assert units and all(not unit.citations for unit in units)
    for unit in units:
        if unit.substantive:
            passed, _, _, _ = judge_unit(unit, pack)
            assert passed is False


async def test_exact_quote_is_verbatim_and_supported(tmp_path: pathlib.Path) -> None:
    index = _fixture_index(tmp_path)
    plan = await run_planner("я бухаю каждый вечер")
    merged = deduplicate_cross_aspect(search_first_round(index, plan))
    pack, _ = load_exact_evidence(
        index, merged, ru_corpus_version=str(index.metadata.get("ru_artifact_sha256", ""))
    )
    first = pack.units[0]
    locator = (
        f"{first.provenance.source_id}/{first.provenance.section_id}#{first.provenance.chunk_id}"
    )
    claim = _first_sentence(first.text)
    answer = f"{claim} [{locator}]"
    diagnostics = build_grounded_response(
        answer=answer,
        pack=pack,
        diagnostics=_diag(),
    )
    substantive = [unit for unit in diagnostics.units if unit.grounding_passed is not None]
    assert substantive and all(unit.grounding_passed for unit in substantive)
    assert any(unit.source_exact for unit in substantive)


def _diag() -> Any:
    from aa.conversation.orchestrator import TurnDiagnostics

    return TurnDiagnostics(
        substantive=True,
        aspects=1,
        retrieval_rounds=1,
        candidates=1,
        evidence_chunks=1,
        evidence_tokens=10,
        tool_call_count=2,
        coverage_gaps=(),
        regeneration_count=0,
        grounding_passed=None,
        served_model=PRIMARY,
        fallback_used=False,
        error_category="ok",
        retry_count=0,
    )


# ---------------------------------------------------------------------------
# Provider contract: bounded retry, technical fallback, no corpus churn
# ---------------------------------------------------------------------------


async def test_transient_503_retries_then_succeeds_on_primary() -> None:
    attempts = 0

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise OpenCodeTransientError("opencode request failed transiently: http=503")
        return "steady answer"

    result = await send_with_fallback(
        _send,
        "ses_1",
        "prompt",
        agent=AGENT_NAME,
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
        sleep=_noop_sleep,
    )
    assert result.text == "steady answer"
    assert result.served_model == PRIMARY
    assert result.fallback_used is False
    assert result.retry_count == 2
    assert attempts == 3


async def test_persistent_503_uses_bounded_technical_fallback_once() -> None:
    seen: list[str] = []

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        seen.append(model)
        if model == PRIMARY:
            raise OpenCodeTransientError("opencode request failed transiently: http=503")
        return "fallback answer"

    result = await send_with_fallback(
        _send,
        "ses_1",
        "prompt",
        agent=AGENT_NAME,
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
        sleep=_noop_sleep,
    )
    assert result.fallback_used is True
    assert result.served_model == FALLBACK
    assert result.error_category == "fallback-used"
    assert seen.count(FALLBACK) == 1


async def test_real_429_always_escapes_synthesis_for_runner_recovery() -> None:
    seen: list[str] = []

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        seen.append(model)
        raise OpenCodeRateLimitError("opencode provider rate-limited: http=429")

    with pytest.raises(OpenCodeRateLimitError):
        await send_with_fallback(
            _send,
            "ses_1",
            "prompt",
            agent=AGENT_NAME,
            primary_model=PRIMARY,
            fallback_model=FALLBACK,
            sleep=_noop_sleep,
        )
    assert seen == [PRIMARY]


async def test_deterministic_provider_error_never_falls_back() -> None:
    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        raise OpenCodeDeterministicError("opencode request rejected: http=400")

    with pytest.raises(TurnFailed) as excinfo:
        await send_with_fallback(
            _send,
            "ses_1",
            "prompt",
            agent=AGENT_NAME,
            primary_model=PRIMARY,
            fallback_model=FALLBACK,
            sleep=_noop_sleep,
        )
    assert excinfo.value.category == "synthesis-failed"


async def test_trivial_turn_uses_named_agent_with_fallback() -> None:
    seen: list[tuple[str, str]] = []

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        seen.append((agent, model))
        assert "по-русски" in prompt
        return "Привет! Чем могу помочь?"

    result = await run_trivial_turn(
        "привет",
        session_id="ses_1",
        send=_send,
        agent=AGENT_NAME,
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
        sleep=_noop_sleep,
    )
    assert result.text == "Привет! Чем могу помочь?"
    assert seen == [(AGENT_NAME, PRIMARY)]


# ---------------------------------------------------------------------------
# Output-limit boundary for downstream #83
# ---------------------------------------------------------------------------


async def test_compact_drops_lower_priority_without_inventing(
    tmp_path: pathlib.Path,
) -> None:
    index = _fixture_index(tmp_path)
    runner = _runner(index)

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        # Two independently cited sentences on separate lines: two units.
        quoted: list[str] = []
        for line in prompt.splitlines():
            match = re.match(
                r"\[([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+#[A-Za-z0-9_:.\-]+)\] (.*)", line
            )
            if match:
                quoted.append(f"{_first_sentence(match.group(2).strip())} [{match.group(1)}]")
            if len(quoted) == 2:
                break
        assert len(quoted) == 2
        return "\n".join(quoted)

    response = await runner.run_grounded_turn(
        "тяга и страх срыва рядом", session_id="ses_1", send=_send
    )
    assert len(response.units) >= 2
    kinds = {unit.kind for unit in response.units}
    assert kinds <= {"exact-quote", "prose", "boilerplate"}
    compacted = response.compact(max_units=1)
    assert len(compacted.units) == 1
    assert compacted.units[0].priority == 0
    assert compacted.units[0].grounding_passed == response.units[0].grounding_passed
    assert compacted.text in response.text
    with pytest.raises(TurnFailed):
        response.compact(max_units=-1)


async def test_synthesis_prompt_never_carries_planner_metadata(
    tmp_path: pathlib.Path,
) -> None:
    index = _fixture_index(tmp_path)
    plan = validate_plan(build_local_plan_payload("я бухаю"), original_query="я бухаю")
    merged = deduplicate_cross_aspect(search_first_round(index, plan))
    pack, _ = load_exact_evidence(
        index, merged, ru_corpus_version=str(index.metadata.get("ru_artifact_sha256", ""))
    )
    prompt = build_synthesis_prompt(user_text="я бухаю", pack=pack)
    assert "ru-query-plan-v1" not in prompt
    assert "lexical_query_en" not in prompt
    assert "preview" not in prompt.lower()


# ---------------------------------------------------------------------------
# Qualification gates + Product Contract #7 dispatch wiring
# ---------------------------------------------------------------------------


def test_runtime_qualification_gate_passes_on_qualified_repo() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/verify_runtime_qualification.py"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status": "qualified"' in completed.stdout


def test_product_contract_dispatch_targets_exact_main_issue7() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    workflow_path = (
        root / ".github" / "workflows" / "aa-product-contract-qualification-dispatch.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")
    assert "branches" in workflow and "main" in workflow
    assert "Verify exact current main identity" in workflow
    assert "refs/remotes/origin/main" in workflow
    # Repository-owned qualification (issue #146): the dispatch relays only
    # to aa-self-proving-qualification.yml with the exact main SHA. The
    # generic continuum-opencode.yml capability/qualification inputs are
    # intentionally absent.
    assert "aa-self-proving-qualification.yml" in workflow
    assert "continuum-opencode.yml" not in workflow
    assert "capability_number=6" not in workflow
    assert "qualification_number=7" not in workflow
    assert "required_sha" in workflow
    assert "verify_product_contract_qualification.py" in workflow
    assert "not-activated" in workflow
    assert "capability_number=9" not in workflow
    assert "qualification_number=40" not in workflow
    assert not (root / ".github" / "workflows" / "aa-issue9-qualification-dispatch.yml").exists()


def test_runtime_model_policy_has_no_third_fallback() -> None:
    from aa.config import Settings

    settings = Settings.from_env({})
    assert settings.opencode_agent == AGENT_NAME
    assert settings.opencode_model == PRIMARY
    assert settings.opencode_fallback_model == FALLBACK
    assert settings.opencode_model != settings.opencode_fallback_model


def test_fail_closed_reply_carries_no_book_claims() -> None:
    # Operational fail-closed message: Russian-only, states inability +
    # asks to refine, carries no citations, no quotations, no factual book
    # assertions and no English fallback text (issue #98).
    assert "уточнить" in FAIL_CLOSED_REPLY
    assert "[" not in FAIL_CLOSED_REPLY
    assert "«" not in FAIL_CLOSED_REPLY
    for marker in ("refine", "cannot", "grounded", "please", "I cannot"):
        assert marker not in FAIL_CLOSED_REPLY
    from aa.conversation.orchestrator import contains_english_fallback, meets_russian_only

    assert not contains_english_fallback(FAIL_CLOSED_REPLY)
    assert meets_russian_only(FAIL_CLOSED_REPLY)
    assert RUNTIME_VERSION.startswith("aa-conversation-runtime/")
