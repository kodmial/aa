"""RU-only user-facing contract regression tests (issue #98).

Every user-visible fallback on the Russian Telegram path must be
deterministic Russian: no OpenCode/provider/internal English error or
fallback text may leak. Uses invented fixture sentences only.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from typing import Any

import pytest

from aa.app import Application
from aa.config import Settings
from aa.conversation.orchestrator import (
    AGENT_NAME,
    TurnFailed,
    TurnRunner,
    build_synthesis_prompt,
    build_trivial_prompt,
    contains_english_fallback,
    deduplicate_cross_aspect,
    load_exact_evidence,
    meets_russian_only,
    run_planner,
    run_trivial_turn,
    search_first_round,
)
from aa.corpus.structure import SECTION_IDS, build_full_structure
from aa.opencode.errors import OpenCodeDeterministicError
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.retrieval.index import HybridIndex, build_hybrid_index

PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"

# Observed live failure shape from issue #98 (paraphrased template, not
# user text): an English insufficient-grounding fallback leaked to Telegram.
OBSERVED_EN_FALLBACK = (
    "cannot provide a grounded answer from the available book excerpts; ask the user to clarify"
)

RU_FIXTURES: dict[str, str] = {
    section: (
        f"Фиктивный отрывок раздела {idx + 1} про трезвость поддержку "
        "сообщества утренние собрания.\n\nВторой абзац раздела продолжается."
    )
    for idx, section in enumerate(SECTION_IDS)
}


def _fixture_index(tmp_path: pathlib.Path) -> HybridIndex:
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": f"Fixture EN {section_id} opening. Second sentence here.",
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
    return build_hybrid_index(
        dict(full),
        ru_manifest={"format": "x", "artifact_sha256": "a" * 64},
        en_manifest={"format": "x", "artifact_sha256": "b" * 64},
        embedding_lock={"model_id": "m", "revision": "r" * 40},
        out_dir=tmp_path / "retrieval",
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


# ---------------------------------------------------------------------------
# Deterministic user-visible fallbacks are Russian-only
# ---------------------------------------------------------------------------


def test_deterministic_fallback_replies_are_russian_only() -> None:
    from aa.app import _NEW_REPLY as new_reply
    from aa.app import _START_REPLY as start_reply
    from aa.app import _TEMPORARY_ERROR_REPLY as temp_reply
    from aa.conversation.failures import SERVICE_ERROR_MARKER

    for reply in (start_reply, new_reply, temp_reply):
        assert re.search(r"[\u0400-\u04ff]", reply) is not None
        assert not contains_english_fallback(reply)
    # The typed service error carries an explicit machine marker so it
    # is never mistaken for AA conversation; protocol replies stay
    # Russian prose.
    assert SERVICE_ERROR_MARKER in temp_reply
    assert meets_russian_only(start_reply)
    assert meets_russian_only(new_reply)


def test_system_prompt_mandates_russian_only_response() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    prompt = (root / "prompts" / "aa-agent-system.md").read_text(encoding="utf-8")
    assert "Always respond in Russian" in prompt
    assert "Never emit English user-facing text" in prompt
    assert "Russian is the primary product language" in prompt


async def _make_pack(tmp_path: pathlib.Path):  # type: ignore[no-untyped-def]
    index = _fixture_index(tmp_path)
    plan = await run_planner("я бухаю каждый вечер")
    merged = deduplicate_cross_aspect(search_first_round(index, plan))
    pack, _ = load_exact_evidence(
        index, merged, ru_corpus_version=str(index.metadata.get("ru_artifact_sha256", ""))
    )
    return pack


async def test_synthesis_prompt_is_ru_only(tmp_path: pathlib.Path) -> None:
    pack = await _make_pack(tmp_path)
    prompt = build_synthesis_prompt(user_text="я бухаю", pack=pack)
    assert "ТОЛЬКО по-русски" in prompt
    assert "английский текст запрещён" in prompt.lower()


def test_trivial_prompt_is_ru_only() -> None:
    prompt = build_trivial_prompt(user_text="привет")
    assert "по-русски" in prompt
    assert "привет" in prompt


# ---------------------------------------------------------------------------
# Language guard unit behavior
# ---------------------------------------------------------------------------


def test_observed_english_fallback_is_detected() -> None:
    assert contains_english_fallback(OBSERVED_EN_FALLBACK) is True
    assert meets_russian_only(OBSERVED_EN_FALLBACK) is False
    assert contains_english_fallback("Could not process the message. Please try again.") is True
    assert contains_english_fallback("opencode request failed transiently: http=429") is True


def test_russian_text_with_citation_passes() -> None:
    text = "Фиктивное утверждение о трезвости [ru-fourth-edition-txt/chapter-2#c1]"
    assert contains_english_fallback(text) is False
    assert meets_russian_only(text) is True


def test_pure_english_greeting_fails_strict_grounded_check() -> None:
    assert meets_russian_only("hello answer") is False


# ---------------------------------------------------------------------------
# Orchestrator: insufficient evidence / grounding failure / regeneration
# ---------------------------------------------------------------------------


async def test_english_synthesis_fallback_regenerates_then_fails_closed_ru(
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
        return OBSERVED_EN_FALLBACK

    with pytest.raises(TurnFailed) as excinfo:
        await runner.run_grounded_turn("я бухаю каждый вечер", session_id="s1", send=_send)
    assert excinfo.value.category in ("language-violation", "grounding-failed")
    assert calls == 2  # one bounded regeneration, then fail closed


async def test_ungrounded_russian_regenerates_once_then_succeeds(
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
        first_line = next(
            line for line in prompt.splitlines() if line.startswith("[") and "] " in line
        )
        locator, text = first_line[1:].split("] ", 1)
        first_sentence = text.split(".")[0].strip()
        return f"{first_sentence}. [{locator}]"

    response = await runner.run_grounded_turn("я бухаю каждый вечер", session_id="s1", send=_send)
    assert calls == 2
    assert response.diagnostics.regeneration_count == 1
    assert meets_russian_only(response.text)


async def test_provider_deterministic_failure_maps_to_ru_fail_closed(
    tmp_path: pathlib.Path,
) -> None:
    index = _fixture_index(tmp_path)
    runner = _runner(index)

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        raise OpenCodeDeterministicError("opencode request rejected: http=400")

    with pytest.raises(TurnFailed) as excinfo:
        await runner.run_grounded_turn("что говорит книга о страхе", session_id="s1", send=_send)
    assert excinfo.value.category == "synthesis-failed"
    assert True


async def test_trivial_english_fallback_fails_closed() -> None:
    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        return "I cannot give a grounded answer from the available book passages."

    with pytest.raises(TurnFailed) as excinfo:
        await run_trivial_turn(
            "привет",
            session_id="s1",
            send=_send,
            agent=AGENT_NAME,
            primary_model=PRIMARY,
            fallback_model=FALLBACK,
            sleep=_noop_sleep,
        )
    assert excinfo.value.category == "language-violation"


# ---------------------------------------------------------------------------
# Application: Russian Telegram turns never surface English fallbacks
# ---------------------------------------------------------------------------


async def test_app_respond_maps_grounding_failure_to_ru_fail_closed() -> None:
    # Issue #301: ordinary turns are generative; when the model stack
    # is unavailable the turn fails as a marked service error (never
    # English mechanics, never synthetic conversation).
    from aa.conversation.failures import is_service_error
    from aa.conversation.output_limits import envelope_passes

    settings = Settings.from_env({})
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )
    app = Application(settings, opencode_runtime=runtime)
    await app.start()
    try:
        reply = await app.respond(42, "я бухаю каждый вечер, что делать")
        assert not contains_english_fallback(reply)
        assert envelope_passes(reply)
        if not is_service_error(reply):
            assert meets_russian_only(reply)
    finally:
        await app.stop()


async def test_app_respond_maps_trivial_english_leak_to_ru_fail_closed() -> None:
    # Issue #301: greeting turns are generative model turns; without a
    # model they fail as a marked service error, never an English leak.
    from aa.conversation.failures import is_service_error
    from aa.conversation.output_limits import envelope_passes

    settings = Settings.from_env({})
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )
    app = Application(settings, opencode_runtime=runtime)
    await app.start()
    try:
        reply = await app.respond(43, "привет")
        assert not contains_english_fallback(reply)
        assert envelope_passes(reply)
        if not is_service_error(reply):
            assert meets_russian_only(reply)
    finally:
        await app.stop()


async def test_app_emergency_russian_turn_is_ru_only() -> None:
    settings = Settings.from_env({})
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )
    app = Application(settings, opencode_runtime=runtime)
    await app.start()
    try:
        reply = await app.respond(7, "Я не могу дышать, задыхаюсь")
        assert "112" in reply
        assert re.search(r"[A-Za-z]", reply) is None
        assert not contains_english_fallback(reply)
    finally:
        await app.stop()
